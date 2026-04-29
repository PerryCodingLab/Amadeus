import torch
import torch.nn as nn
import torch.nn.functional as F

class MIDISlidingWindowDataset(Dataset):
    def __init__(self, token_sequences, seq_len, stride=1):
        """
        Args:
            token_sequences: List of 1D integer lists (one list per MIDI file/track)
            seq_len: The size of the context window (e.g., 128 tokens)
            stride: How many tokens to step forward for the next window
        """
        self.seq_len = seq_len
        self.inputs = []
        self.targets = []
        
        # Build the overlapping windows for every MIDI sequence
        for seq in token_sequences:
            # We need at least seq_len + 1 tokens to make an input/target pair
            if len(seq) <= seq_len:
                continue
                
            # Slide the window across the sequence
            for i in range(0, len(seq) - seq_len, stride):
                # Input window: [i ... i + seq_len - 1]
                input_chunk = seq[i : i + seq_len]
                
                # Target window: shifted by 1 token -> [i+1 ... i + seq_len]
                target_chunk = seq[i + 1 : i + seq_len + 1]
                
                self.inputs.append(input_chunk)
                self.targets.append(target_chunk)
                
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        # Convert lists to PyTorch LongTensors (required for CrossEntropyLoss)
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Linear layers to project the queries, keys, and values
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, current_hidden, encoder_outputs):
        # current_hidden: (batch, 1, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)
        Q = self.query_proj(current_hidden)
        K = self.key_proj(encoder_outputs)
        V = self.value_proj(encoder_outputs)
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(attention_weights, V)
        return context, attention_weights


class ComposerMIDI(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super(ComposerMIDI, self).__init__()
        self.hidden_size = hidden_size
        
        # Map MIDI tokens to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # The core sequential memory
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Attention layer to reference themes from the prompt
        self.attention = Attention(hidden_size)
        
        # Final output projection to next-token probabilities
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply attention. We use the current step's output to query the sequence
        # (In a real autoregressive loop, encoder_outputs would be cached)
        context, attn_weights = self.attention(lstm_out, lstm_out)
        
        # Concatenate LSTM output and Attention context
        combined = torch.cat((lstm_out, context), dim=2)
        
        # Predict next token (Cross-Entropy expects unnormalized logits)
        logits = self.fc_out(combined)
        return logits, hidden