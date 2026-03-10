import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.GenreClassifier import MusicGenreClassifier

# --- Configuration ---
CONFIG = {
    'vocab_size': 300,        # Size of REMI vocabulary (adjust based on tokenization)
    'num_classes': 4,         # e.g., Rock, Jazz, Pop, Classical
    'd_model': 128,
    'batch_size': 32,
    'seq_len': 512,
    'epochs': 20,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- Dataset Placeholder ---
class GenreDataset(Dataset):
    """
    Loads tokenized chunks and genre labels.
    Expected Input: Pre-processed JSON or Tensor files.
    """
    def __init__(self, num_samples=1000, seq_len=512):
        # TODO: Replace this with real file loading logic
        # For now, we generate random data to test the pipeline
        self.data = torch.randint(0, CONFIG['vocab_size'], (num_samples, seq_len))
        self.labels = torch.randint(0, CONFIG['num_classes'], (num_samples,))

        # IMPORTANT: Prepend [CLS] token (ID = 1 for example) to every sequence
        cls_tokens = torch.ones((num_samples, 1), dtype=torch.long)
        self.data = torch.cat((cls_tokens, self.data[:, :-1]), dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Training Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        # 1. Zero Gradients
        optimizer.zero_grad()
        # 2. Forward Pass
        output = model(data)
        # 3. Calculate Loss
        loss = criterion(output, target)
        # 4. Backward Pass
        loss.backward()
        # 5. Optimizer Step
        optimizer.step()
        # Metrics tracking
        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy
    
#--- Main Execution---
if __name__ == "__main__":
    print(f"Training on device: {CONFIG['device']}")
    # 1. Prepare Data
    # Note: In a real scenario, you handle the song-split logic in your dataset creation script
    dataset = GenreDataset(num_samples=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    # 2. Initialize Model
    model = MusicGenreClassifier(vocab_size=CONFIG['vocab_size'],
                                num_classes=CONFIG['num_classes'],
                                d_model=CONFIG['d_model'],
                                num_layers=4, nheads=4).to(CONFIG['device'])
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 4. Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_classifier.pth")
            print("   -> Saved best model!")