import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Transformers process all data at once, not sequentially. 
    If you give it the notes [C, E, G], it doesn't know 'C' came first. 
    This class creates a mathematical 'timestamp' (using sine and cosine waves) 
    and attaches it to each note so the model understands the temporal order.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        # 'd_model' is the size of the embedding vector (e.g., 128).
        # 'max_len' is the maximum length of a song chunk we expect (e.g., 5000 tokens).
        super().__init__()
        
        # 1. Create an empty matrix of shape [Max_Length, Vector_Size]
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a column of position indices (0, 1, 2, 3...)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Calculate the frequencies for the sine/cosine waves
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 4. Apply Sine to even indices and Cosine to odd indices of the vector
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. Register as a 'buffer'. 
        # In PyTorch, a buffer is a variable that is saved with the model but is NOT 
        # updated by the optimizer during training (it's a fixed mathematical rule).
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # When data (x) passes through here, we simply add the positional wave 
        # on top of the note embeddings.
        x = x + self.pe[:, :x.size(1), :]
        return x

class MusicGenreClassifier(nn.Module):
    """
    THEORY: This is the main neural network. It is a Transformer Encoder, 
    often called a 'BERT-style' model. It reads a sequence of MIDI tokens, 
    figures out how the notes relate to each other (Self-Attention), and predicts the genre.
    """
    def __init__(self,
                vocab_size: int,         # How many unique MIDI tokens exist (e.g., 300)
                num_classes: int,        # How many genres we are classifying (e.g., 4)
                d_model: int = 128,      # How big the vector for each note is
                nhead: int = 4,          # How many 'attention heads' look at the data
                num_layers: int = 2,     # How many Transformer blocks are stacked on each other
                dim_feedforward: int = 512, # Size of the hidden layer inside the transformer
                dropout: float = 0.1):   # % of neurons to randomly turn off to prevent memorization
        super().__init__()
        
        # Save d_model to the class so we can use it in the forward pass
        self.d_model = d_model
        
        # 1. EMBEDDING LAYER
        # Converts simple integer IDs (like token #45) into dense arrays of numbers (vectors).
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. POSITIONAL ENCODING
        # Injects the time-stamp waves we defined in the class above.
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. TRANSFORMER ENCODER BLOCKS
        # This defines a single "layer" of the transformer.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True) # batch_first means data shape is (Batch, Sequence, Features)
        
        # This stacks the layers on top of each other.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. CLASSIFICATION HEAD
        # After the transformer understands the music, this final standard neural network 
        # shrinks the 128-sized vector down to the number of genres (e.g., 4) to make a guess.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), # Activation function to introduce non-linearity
            nn.Linear(d_model, num_classes)
        )

        # Initialize the random starting weights of the network
        self.init_weights()
    
    def init_weights(self):
        """
        Neural networks start with random weights. Setting them to specific 
        small, uniform numbers helps the model learn faster and prevents math errors early on.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier[2].bias.data.zero_()
        self.classifier[2].weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, mask=None):
        """
        The forward pass dictates exactly how data flows through the model 
        from start to finish when we make a prediction.
        """
        # 1. Convert integer tokens to embeddings, and scale them up mathematically.
        # Scaling by the square root of the vector size is standard practice in Transformers 
        # to keep the variance stable before adding positional waves.
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # 2. Add the time-stamps.
        src = self.pos_encoder(src)
        
        # 3. Pass through the "brain" (Self-Attention layers).
        output = self.transformer_encoder(src, mask)
        
        # 4. Extract the [CLS] Token.
        # 'output' contains the processed vectors for EVERY note. 
        # We only care about the very first token (Index 0), which is a special [CLS] (Classification) 
        # token designed to summarize the whole song.
        cls_output = output[:, 0, :]
        
        # 5. Make the final genre prediction.
        output = self.classifier(cls_output)
        
        return output