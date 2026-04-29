from pathlib import Path
import torch
MIDI_DATA_DIR = Path("Data/RawData")
OUT_DIR = Path("Data/ProccessedData")
TOKENIZER_PATH = Path("Data/tokenizers")
TOKENIZER_PARAMS = {}

GenreClassifierCONFIG = {
    'vocab_size': 300,        # Size of REMI vocabulary (adjust based on tokenization)
    'num_classes': 32,         # e.g., Rock, Jazz, Pop, Classical
    'd_model': 128,
    'batch_size': 32,
    'seq_len': 512,
    'epochs': 20,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# GenreClassifierCONFIG = {
#     'vocab_size': 300,        # Size of REMI vocabulary (adjust based on tokenization)
#     'num_classes': 32,         # e.g., Rock, Jazz, Pop, Classical
#     'd_model': 128,
#     'batch_size': 32,
#     'seq_len': 512,
#     'epochs': 20,
#     'learning_rate': 1e-4,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu'
# }