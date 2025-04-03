import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import MedicalTrialClassifier
from train import MedicalTrialDataset
from torch.utils.data import DataLoader
import torch
import os

def evaluate_model(model_path: str, test_data_path: str):
    """Evaluate the model and generate performance metrics."""
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    # Load model
    print("Loading model...")
    model = MedicalTrialClassifier()
    try:
        model.load(model_path)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    # Load and validate test data
    print("Loading test data...")
    try:
        df = pd.read_csv(test_data_path)
        
        # Check for NaN values
        if df['description'].isna().any() or df['label'].isna().any():
            print("Warning: Found NaN values in data")
            print(f"NaN in description: {df['description'].isna().sum()}")
            print(f"NaN in label: {df['label'].isna().sum()}")
            
            # Remove rows with NaN values
            df = df.dropna(subset=['description', 'label'])
            print(f"Rows remaining after removing NaN: {len(df)}")
        
        # Validate labels
        valid_labels = set(model.label2id.keys())
        invalid_labels = set(df['label'].unique()) - valid_labels
        if invalid_labels:
            print(f"Warning: Found invalid labels: {invalid_labels}")
            df = df[df['label'].isin(valid_labels)]
            print(f"Rows remaining after removing invalid labels: {len(df)}")
        
        # Convert labels to IDs
        df['label_id'] = df['label'].map(model.label2id)
        
    except Exception as e:
        raise Exception(f"Error loading test data: {str(e)}")

    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")

    # Create test dataset
    test_dataset = MedicalTrialDataset(
        df['description'].values,
        df['label_id'].values,
        model.tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Get predictions
    print("Making predictions...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label']
            
            logits = model.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1).cpu()
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    # Convert to labels
    pred_labels = [model.id2label[p] for p in all_predictions]
    true_labels = [model.id2label[l] for l in all_labels]

    # Generate classification report
    report = classification_report(true_labels, pred_labels)
    print("\nClassification Report:")
    print(report)

    # Generate confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        xticklabels=model.label2id.keys(),
        yticklabels=model.label2id.keys(),
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Save detailed results
    results = {
        'true_labels': true_labels,
        'predicted_labels': pred_labels,
        'classification_report': report,
    }
    
    print("\nEvaluation complete. Results saved to confusion_matrix.png")
    return results

if __name__ == "__main__":
    try:
        results = evaluate_model('best_model.pt', '../data/trials.csv')
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise 