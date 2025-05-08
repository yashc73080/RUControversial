import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dotenv import load_dotenv
from sqlalchemy import create_engine

from tqdm import tqdm
import os
import re
import numpy as np
import random
import json


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BERTClassifier(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-small', num_classes=4, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        
        # Add intermediate layer
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass token_type_ids if model supports it
        if token_type_ids is not None and 'bert-' in self.bert.config._name_or_path:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the [CLS] token representation instead of mean pooling
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply intermediate layer
        x = self.intermediate(pooled_output)
        
        # Apply classifier
        logits = self.classifier(x)
        
        return logits
    
    def train_model(self, train_loader, optimizer, scheduler, loss_fn, device, epochs=3, eval_loader=None, patience=3):
        best_f1 = 0
        patience_counter = 0
        
        # Initialize lists to store metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            total_loss = 0
            total_steps = 0
            epoch_predictions = []
            epoch_true_labels = []
            
            # Only unfreeze BERT parameters after first epoch
            if epoch == 1:
                print("Unfreezing encoder parameters...")
                for param in self.bert.parameters():
                    param.requires_grad = True
            
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Handle token_type_ids if present
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    outputs = self(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self(input_ids, attention_mask)
                
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Track predictions for accuracy calculation
                _, preds = torch.max(outputs, dim=1)
                epoch_predictions.extend(preds.cpu().numpy())
                epoch_true_labels.extend(labels.cpu().numpy())
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_steps += 1

                loop.set_description(f'Epoch {epoch+1}/{epochs}')
                loop.set_postfix(loss=loss.item())
            
            # Calculate training metrics for this epoch
            avg_train_loss = total_loss / total_steps
            train_acc = accuracy_score(epoch_true_labels, epoch_predictions)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Avg training loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Evaluation phase
            if eval_loader:
                # Track validation loss
                self.eval()
                val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for batch in eval_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        token_type_ids = batch.get('token_type_ids')
                        if token_type_ids is not None:
                            token_type_ids = token_type_ids.to(device)
                            outputs = self(input_ids, attention_mask, token_type_ids)
                        else:
                            outputs = self(input_ids, attention_mask)
                        
                        loss = loss_fn(outputs, labels)
                        val_loss += loss.item()
                        val_steps += 1
                
                # Calculate validation metrics
                avg_val_loss = val_loss / val_steps
                val_losses.append(avg_val_loss)
                
                # Get standard evaluation metrics
                _, _, eval_acc, eval_f1 = self.evaluate_model(eval_loader, device)
                val_accuracies.append(eval_acc)
                val_f1_scores.append(eval_f1)
                
                print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
                
                # Early stopping logic
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    patience_counter = 0
                    # Save the best model
                    torch.save(self.state_dict(), "best_bert_model.pth")
                    print(f"New best model saved with F1: {best_f1:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Load the best model
                        self.load_state_dict(torch.load("best_bert_model.pth"))
                        break
        
        # Return all collected metrics
        metrics = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores
        }
        return metrics

    def evaluate_model(self, test_loader, device):
        self.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Handle token_type_ids if present
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    outputs = self(input_ids, attention_mask, token_type_ids)
                else:
                    outputs = self(input_ids, attention_mask)
                
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return predictions, true_labels, acc, f1
    
    
class RedditDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        focal_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return focal_loss.mean()

def preprocess_text(text):
    """Clean and preprocess text data."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Reddit formatting
        text = re.sub(r'\[.*?\]', '', text)  # Remove [AITA] tags
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def get_df():
    load_dotenv()

    # Fetch variables
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    # Connect to the database
    try:
        engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}")
        print("Connection successful!")

    except Exception as e:
        print(f"Failed to connect: {e}")

    # Get pandas df
    query = "SELECT id, combined_text, verdict FROM aita_posts;"
    df = pd.read_sql_query(query, engine)
    
    # Clean the data
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    
    # Remove rows with empty text or missing values
    df = df.dropna(subset=['combined_text', 'verdict'])
    df = df[df['combined_text'].str.len() > 10]  # Remove very short posts
    
    return df

def prepare_dataloaders(df, model_name='roberta-base', max_length=256, val_size=0.1):
    label_encoder = LabelEncoder()
    df['verdict_encoded'] = label_encoder.fit_transform(df['verdict'])
    
    # Split into train, validation, and test sets
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        df['combined_text'].values, 
        df['verdict_encoded'].values, 
        test_size=0.2, 
        random_state=42,
        stratify=df['verdict_encoded'].values  # Stratified split to maintain class distribution
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=val_size/(1-0.2),  # Adjust validation size 
        random_state=42,
        stratify=train_val_labels  # Stratified split
    )

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_encodings = tokenizer(
        list(train_texts), 
        truncation=True, 
        padding='max_length', 
        max_length=max_length, 
        return_tensors='pt'
    )
    
    val_encodings = tokenizer(
        list(val_texts), 
        truncation=True, 
        padding='max_length', 
        max_length=max_length, 
        return_tensors='pt'
    )
    
    test_encodings = tokenizer(
        list(test_texts), 
        truncation=True, 
        padding='max_length', 
        max_length=max_length, 
        return_tensors='pt'
    )

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)

    # Create datasets
    train_dataset = RedditDataset(train_encodings, train_labels)
    val_dataset = RedditDataset(val_encodings, val_labels)
    test_dataset = RedditDataset(test_encodings, test_labels)

    # Direct weight assignment based on observed performance
    class_weights = torch.tensor([
        8.0,  # "asshole"
        40.0, # "everyone sucks" - highest weight due to poorest detection
        30.0, # "no assholes here"
        1.5   # "not the asshole"
    ], dtype=torch.float)
    
    # Create DataLoaders
    train_loader = DataLoader( train_dataset, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = DataLoader( val_dataset, batch_size=16, shuffle=False, num_workers=4 )
    test_loader = DataLoader( test_dataset, batch_size=16, shuffle=False, num_workers=4 )

    print('Data preparation complete.')
    print(f'Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}')
    print(f'Class distribution in training set: {np.bincount(train_labels.numpy())}')
    print(f'Class weights: {class_weights.numpy()}')

    return train_loader, val_loader, test_loader, label_encoder, class_weights

def load_bert_model(model_path="../final_aita_model.pth"):
    """Load and initialize the BERT model"""
    # Load the saved model with map_location to ensure compatibility with CPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model_name = checkpoint.get('model_name', 'roberta-base')
    label_encoder = checkpoint['label_encoder']
    
    # Initialize model
    model = BERTClassifier(model_name=model_name, num_classes=len(label_encoder.classes_))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded model:', model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Also load and initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'device': device
    }

def predict_with_model(text, loaded_model):
    """Make predictions with already loaded model"""
    model = loaded_model['model']
    tokenizer = loaded_model['tokenizer']
    label_encoder = loaded_model['label_encoder']
    device = loaded_model['device']
    
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize
    encodings = tokenizer(
        preprocessed_text, 
        truncation=True, 
        padding='max_length', 
        max_length=256, 
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Handle token_type_ids if present
    token_type_ids = encodings.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        outputs = model(input_ids, attention_mask, token_type_ids)
    else:
        outputs = model(input_ids, attention_mask)
    
    # Get prediction
    print('Predicting...')
    probs = F.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    
    verdict = label_encoder.inverse_transform([prediction.item()])[0]
    confidence = confidence.item()
    
    # Get all class probabilities
    all_probs = probs[0].cpu().detach().numpy()
    class_probs = {label_encoder.inverse_transform([i])[0]: float(prob) 
                  for i, prob in enumerate(all_probs)}
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'class_probabilities': class_probs
    }

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Load the DataFrame
    df = get_df()
    
    # Print dataset statistics
    print(f"Total number of samples: {len(df)}")
    print(f"Class distribution: {df['verdict'].value_counts()}")
    
    # Choose model
    model_name = 'microsoft/deberta-v3-small'
    
    # Get the dataloaders with validation set
    train_loader, val_loader, test_loader, label_encoder, class_weights = prepare_dataloaders(
        df, model_name=model_name, max_length=256
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize the model
    model = BERTClassifier(model_name=model_name, num_classes=len(label_encoder.classes_))
    model.to(device)

    # Freeze bert parameters initially (for first epoch)
    for param in model.bert.parameters():
        param.requires_grad = False

    # Initialize optimizer with weight decay
    epochs = 5  # Reduce epochs with early stopping
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.15 * total_steps)
    
    # Use different learning rates for different components
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': 5e-6, 'weight_decay': 0.01},
        {'params': [p for n, p in model.intermediate.named_parameters()], 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': 5e-5, 'weight_decay': 0.01}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Use class weights for loss function
    class_weights = class_weights.to(device)
    loss_fn = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    patience = 3  # Early stopping patience

    print("Model and optimizer initialized.")
    print(f"Training with {epochs} epochs, early stopping with patience={patience}.")

    # Train the model with early stopping
    training_metrics = model.train_model(
        train_loader, 
        optimizer, 
        scheduler, 
        loss_fn, 
        device, 
        epochs=epochs, 
        eval_loader=val_loader,
        patience=patience
    )
    
    print("Training complete.")

    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_bert_model.pth"))
    
    # Evaluate the model
    predictions, true_labels, acc, f1 = model.evaluate_model(test_loader, device)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    # Save the final model
    model.cpu()
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'model_name': model_name
    }, "final_aita_model.pth")
    print("Final model saved to final_aita_model.pth")

    metrics_data = {
        'predictions': [int(x) for x in predictions],
        'true_labels': [int(x) for x in true_labels],
        'test_accuracy': float(acc),
        'test_f1': float(f1),
        'training_metrics': {
            'train_losses': [float(x) for x in training_metrics['train_losses']],
            'train_accuracies': [float(x) for x in training_metrics['train_accuracies']],
            'val_losses': [float(x) for x in training_metrics['val_losses']],
            'val_accuracies': [float(x) for x in training_metrics['val_accuracies']],
            'val_f1_scores': [float(x) for x in training_metrics['val_f1_scores']]
        }
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("Metrics saved to model_metrics.json")

if __name__ == "__main__":
    main()