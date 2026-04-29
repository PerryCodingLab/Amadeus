import torch.optim as optim

def train_composer(model, dataloader, epochs, vocab_size, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # inputs: [batch, seq_len]
            # targets: [batch, seq_len] (shifted by 1 from inputs)
            
            optimizer.zero_grad()
            
            logits, _ = model(inputs)
            
            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            # Gradient clipping is highly recommended for LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# --- Evaluation Metrics ---

def calculate_inception_score(generated_midi_tokens):
    """
    Placeholder for Inception Score calculation.
    Requires an auxiliary pre-trained classification model to assess diversity.
    """
    pass

def check_musical_coherence(generated_midi_tokens):
    """
    Algorithmic checks for scale and rhythm.
    e.g., checking if generated notes fall within the inferred key,
    or if the time-deltas align with standard rhythmic subdivisions.
    """
    coherence_score = 0.0 
    # TODO: Implement token-parsing logic to check intervals and timing
    return coherence_score