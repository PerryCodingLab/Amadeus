import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from miditok.pytorch_data import DataCollator
from miditok import REMI
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath("."))
from models.GenreClassifier import MusicGenreClassifier
from src.utils import suppress_c_stdout
import time
from tqdm import tqdm 
from src.config import GenreClassifierCONFIG as CONFIG
# --- Configuration ---


# --- Dataset Placeholder ---
class GenreDataset(Dataset):
    """
    Loads tokenized chunks and genre labels.
    Expected Input: Pre-processed JSON or Tensor files.
    """
    def __init__(self, tokens_path, labels_path):
        # TODO: Replace this with real file loading logic
        # For now, we generate random data to test the pipeline
        self.tokens = torch.load(tokens_path, weights_only=False)
        self.labels = torch.load(labels_path, weights_only=False)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # return {
        #     "input_ids": self.tokens[idx],
        #     "labels": self.labels[idx]
        # }
        return (
            torch.tensor(self.tokens[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# --- Training Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # for x in enumerate(dataloader):
    #     print("and x is?: ", x)
    # for batch_idx, batch in enumerate(dataloader):
    #     data = batch['input_ids'].to(device)
    #     target = batch['labels'].squeeze(-1).to(device)
    # print(type(dataloader.tokens[0]))
    # print(dataloader.tokens[0][:10])
    for data, target in tqdm(dataloader):
        data = data.to(device)
        target = target.squeeze(-1).to(device)
        # data, target = data.to(device), target.to(device)
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
            data = data.to(device)
            target = target.squeeze(-1).to(device)
            # data, target = data.to(device), target.to(device)
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
    date = time.perf_counter()
    print(f"Training on device: {CONFIG['device']}")
    tokenizer = REMI()
    train_dataset = GenreDataset("Data/ProccessedData/train_tokens.pt","Data/ProccessedData/train_labels.pt")
    val_dataset = GenreDataset("Data/ProccessedData/test_tokens.pt","Data/ProccessedData/test_labels.pt")
    # collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=False, shift_labels=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True) # collate_fn=collator
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False) # collate_fn=collator
    # 2. Initialize Model
    model = MusicGenreClassifier(vocab_size=CONFIG['vocab_size'],
                                num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 4. Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG['epochs']):
        start = time.perf_counter()
        # with suppress_c_stdout():
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            classifiername = str(date) + "best_classifier.pth"
            torch.save(model.state_dict(), classifiername)
            print("   -> Saved best model!")


def trainOnDataset(train_dataset, val_dataset, tokenizer):
    print(f"Training on device: {CONFIG['device']}")

    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=False, shift_labels=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collator)
    # 2. Initialize Model
    model = MusicGenreClassifier(vocab_size=CONFIG['vocab_size'],
                                num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    
    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 4. Training Loop
    best_val_acc = 0.0
    for epoch in range(CONFIG['epochs']):
        start = time.perf_counter()
        with suppress_c_stdout():
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_classifier.pth")
            print("   -> Saved best model!")


# if __name__ == "__main__":
#     print(f"Training on device: {CONFIG['device']}")
#     # 1. Prepare Data
#     # Note: In a real scenario, you handle the song-split logic in your dataset creation script
#     dataset = GenreDataset(num_samples=1000)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
#     # 2. Initialize Model
#     model = MusicGenreClassifier(vocab_size=CONFIG['vocab_size'],
#                                 num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    
#     # 3. Define Loss and Optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

#     # 4. Training Loop
#     best_val_acc = 0.0
#     for epoch in range(CONFIG['epochs']):
#         train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
#         val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

#         print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
#         print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
#         print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_classifier.pth")
#             print("   -> Saved best model!")