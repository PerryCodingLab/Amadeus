from miditok import REMI, TokenizerConfig
from pathlib import Path
from configs.config import TOKENIZER_PATH, TOKENIZER_PARAMS 
import sys


name = 'temp'
print(len(sys.argv))
if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    print("No command-line arguments provided for tokenizer name, saving to temp.json")
name = name + '.json'


tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))

tokenizer.save(TOKENIZER_PATH / name)