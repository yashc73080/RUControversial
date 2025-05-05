import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from dotenv import load_dotenv
from sqlalchemy import create_engine

from tqdm import tqdm
import os


class BERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        x = self.dropout(pooled_output)

        return self.classifier(x)
    
    def train_model(self, train_loader, optimizer, scheduler, loss_fn, device, epochs=3):
        self.train()
        for epoch in range(epochs): 
            # Unfreeze BERT parameters after the first epoch
            if epoch == 1:
                for param in self.bert.parameters():
                    param.requires_grad = True

            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

    def evaluate_model(self, test_loader, device):
        self.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self(input_ids, attention_mask)
                preds = torch.argmax(outputs, axis=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Metrics
        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f'Accuracy: {acc:.4f}')
        print(f'F1-Score: {f1:.4f}')

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

    return df

def prepare_dataloaders(df):
    label_encoder = LabelEncoder()
    df['verdict_encoded'] = label_encoder.fit_transform(df['verdict'])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['combined_text'].values, 
        df['verdict_encoded'].values, 
        test_size=0.2, 
        random_state=42
    )

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    train_dataset = RedditDataset(train_encodings, train_labels)
    test_dataset = RedditDataset(test_encodings, test_labels)

    # Create DataLoaders
    label_array = train_labels.numpy()
    class_sample_counts = np.bincount(label_array)
    weights = 1.0 / class_sample_counts # inverse frequency
    sample_weights = weights[label_array]
    sampler = WeightedRandomSampler( # Account for class imbalance
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, test_loader, label_encoder, train_labels, test_labels


def main():
    
    # Load the DataFrame
    df = get_df()

    # Get the dataloaders
    train_loader, test_loader, label_encoder, train_labels, test_labels = prepare_dataloaders(df)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the model
    model = BERTClassifier()
    model.to(device)

    # Get unique classes and compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels.numpy()),
        y=train_labels.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize optimizer, loss function, and scheduler
    epochs = 8
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Train the model
    for param in model.bert.parameters(): # Freeze BERT parameters initially
        param.requires_grad = False

    model.train_model(train_loader, optimizer, scheduler, loss_fn, device, epochs=8)

    # Evaluate the model
    predictions, true_labels, acc, f1 = model.evaluate_model(test_loader, device)
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    # Save the model
    model.cpu()
    torch.save(model.state_dict(), "bert_model.pth")
    print("Model saved to bert_model.pth")

if __name__ == "__main__":
    main()