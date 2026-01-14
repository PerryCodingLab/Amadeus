from miditok import REMI
from pathlib import Path
from configs.config import TOKENIZER_PATH, MIDI_DATA_DIR, OUT_DIR
from tqdm import tqdm #only for visual
import sys


def tokenizeData(tokenizerName):
    tokenizer = REMI(params=Path(TOKENIZER_PATH / tokenizerName))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    midi_files = Path(MIDI_DATA_DIR).glob("**/*.mid")
    # print(len(midi_files))
    count = 0
    for midi_path in tqdm(midi_files):
        tokens = tokenizer(midi_path)
        out_path = Path(OUT_DIR, midi_path.stem + ".json")
        tokenizer.save_tokens(tokens, out_path)
        count+=1
        if count == 5:
            return



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No command-line arguments provided for the tokenizer name.")
        sys.exit(1)
    tokenizeData(sys.argv[1] + '.json')
    

