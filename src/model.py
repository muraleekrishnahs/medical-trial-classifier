import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import nn
from typing import List, Dict, Any
import joblib

class MedicalTrialClassifier:
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 5)  # 5 classes
        self.bert.to(self.device)
        self.classifier.to(self.device)
        
        self.label2id = {
            "Dementia": 0,
            "ALS": 1,
            "Obsessive Compulsive Disorder": 2,
            "Scoliosis": 3,
            "Parkinson's Disease": 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def eval(self):
        """Set the model to evaluation mode"""
        self.bert.eval()
        self.classifier.eval()

    def train(self, mode=True):
        """Set the model to training mode"""
        self.bert.train(mode)
        self.classifier.train(mode)

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare text for the model"""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, text: str) -> str:
        """Predict the label for a given text"""
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            inputs = self.preprocess(text)
            logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
            predicted_class = torch.argmax(logits, dim=1).item()
            
        return self.id2label[predicted_class]

    def save(self, path: str):
        """Save the model"""
        torch.save({
            'bert_state_dict': self.bert.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'label2id': self.label2id,
        }, path)

    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.bert.load_state_dict(checkpoint['bert_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.label2id = checkpoint['label2id']
        self.id2label = {v: k for k, v in self.label2id.items()} 