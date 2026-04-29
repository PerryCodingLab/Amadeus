import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import optuna
import pandas as pd
import matplotlib.pyplot as plt

# Import your model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.GenreClassifier import MusicGenreClassifier
from training.train_classifier import GenreDataset
from tqdm import tqdm
# --- Dataset Placeholder (Replace with real data later) ---
class DummyGenreDataset(Dataset):
    def __init__(self, num_samples=500, seq_len=512):
        self.data = torch.randint(0, 300, (num_samples, seq_len))
        self.labels = torch.randint(0, 4, (num_samples,))
        cls_tokens = torch.ones((num_samples, 1), dtype=torch.long)
        self.data = torch.cat((cls_tokens, self.data[:, :-1]), dim=1)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- Core Tuning Logic ---
def objective(trial):
    """
    Optuna will run this function multiple times. Every 'trial' it will 
    pick a new combination of hyperparameters to test.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Let Optuna suggest hyperparameters!
    # We constrain nhead so it always perfectly divides d_model (a PyTorch requirement)
    d_model = trial.suggest_categorical('d_model', [64, 128, 256]) # Baseline is 128 
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])          # Baseline is 4 
    num_layers = trial.suggest_int('num_layers', 2, 6)             # Baseline is 2-4 
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    # Initialize model with these specific suggestions
    model = MusicGenreClassifier(
        vocab_size=300, 
        num_classes=32, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers, 
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Load Data (using a smaller batch size to prevent memory errors during tests)
    train_set = GenreDataset("Data/ProccessedData/train_tokens.pt","Data/ProccessedData/train_labels.pt")
    val_set = GenreDataset("Data/ProccessedData/test_tokens.pt","Data/ProccessedData/test_labels.pt")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # 2. Train for a few epochs to see if this architecture has potential
    epochs = 5 
    best_acc = 0.0

    for epoch in range(epochs):
        print("trial:", trial.number, ", Epoch: ",epoch)
        sys.stdout.flush()
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.squeeze(-1).to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        best_acc = max(best_acc, acc)

        # Tell Optuna how we are doing. If the score is terrible, Optuna can "prune" (kill) the trial early.
        print("trial:", trial.number, ", Epoch: ",epoch ,", acc:",acc)
        sys.stdout.flush()
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    print("parameters: ",d_model, nhead,num_layers,dropout,lr)
    sys.stdout.flush()
    return best_acc

def save_visual_reports(study):
    """Saves the experiment results to a CSV table and a PNG graph."""
    os.makedirs("reports", exist_ok=True)
    
    # 1. Save as a Table (CSV)
    df = study.trials_dataframe()
    # Filter out confusing internal Optuna columns for a clean table
    clean_df = df[['number', 'value', 'params_d_model', 'params_nhead', 'params_num_layers', 'params_dropout', 'params_learning_rate', 'state']]
    clean_df.to_csv("reports/tuning_results.csv", index=False)
    print("-> Saved detailed table to reports/tuning_results.csv")

    # 2. Generate a Bar Graph of the Top 5 Trials
    completed_trials = clean_df[clean_df['state'] == 'COMPLETE']
    top_5 = completed_trials.nlargest(5, 'value')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f"Trial {i}" for i in top_5['number']], top_5['value'], color='skyblue')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Accuracy')
    plt.title('Top 5 Transformer Architectures')
    plt.ylim(0, 1.0) # Accuracy goes from 0 to 1
    
    # Add parameter text to the graph for easy reading
    for bar, (_, row) in zip(bars, top_5.iterrows()):
        text = f"L:{row['params_num_layers']}\nH:{row['params_nhead']}\nD:{row['params_d_model']}"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15, text, ha='center', color='black', fontsize=9)

    plt.savefig("reports/top_models_graph.png")
    print("-> Saved visual graph to reports/top_models_graph.png")

if __name__ == "__main__":
    print("Starting Architecture Tuning with Optuna...")
    # Create an Optuna study. We want to 'maximize' the accuracy.
    study = optuna.create_study(direction="maximize")
    print("optimizing: ")
    # Run 10 different architectural trials (increase to 50+ when using real data)
    study.optimize(objective, n_trials=2)

    print("\n=== TUNING COMPLETE ===")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Generate our artifacts
    save_visual_reports(study)