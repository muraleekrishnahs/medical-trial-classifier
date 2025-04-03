import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import MedicalTrialClassifier

class MedicalTrialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }

def prepare_data(df, model):
    """Prepare and validate the data for training."""
    print("Validating and cleaning data...")
    
    # Check for NaN values
    if df['description'].isna().any() or df['label'].isna().any():
        print("Found NaN values in data")
        print(f"NaN in description: {df['description'].isna().sum()}")
        print(f"NaN in label: {df['label'].isna().sum()}")
        
        # Remove rows with NaN values
        df = df.dropna(subset=['description', 'label'])
        print(f"Rows remaining after removing NaN: {len(df)}")
    
    # Validate labels
    valid_labels = set(model.label2id.keys())
    invalid_labels = set(df['label'].unique()) - valid_labels
    if invalid_labels:
        print(f"Found invalid labels: {invalid_labels}")
        df = df[df['label'].isin(valid_labels)]
        print(f"Rows remaining after removing invalid labels: {len(df)}")
    
    # Convert labels to IDs
    df['label_id'] = df['label'].map(model.label2id)
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    return df

def train_model(model, train_loader, val_loader, epochs=3, learning_rate=2e-5):
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate * 10}
    ])
    criterion = nn.CrossEntropyLoss()
    
    device = model.device
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model.forward(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model.forward(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print(f'Validation accuracy: {accuracy:.2f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f'Saving best model with validation loss: {avg_val_loss:.4f}')
            model.save('best_model.pt')

def main():
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv('../data/trials.csv')
    
    # Create model instance
    model = MedicalTrialClassifier()
    
    # Prepare and validate data
    df = prepare_data(df, model)
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['description'].values,
        df['label_id'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label_id'].values
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MedicalTrialDataset(train_texts, train_labels, model.tokenizer)
    val_dataset = MedicalTrialDataset(val_texts, val_labels, model.tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 