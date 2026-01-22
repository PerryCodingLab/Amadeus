from miditok import REMI
from pathlib import Path
from processing.MidiDataSet import getMidiByMetaData
from configs.config import TOKENIZER_PATH, MIDI_DATA_DIR, OUT_DIR
from tqdm import tqdm #only for visual
import sys


def tokenizeData(tokenizerName, outputFolderName='temp', genre_keywords=None):
    tokenizer = REMI(params=Path(TOKENIZER_PATH / tokenizerName))
    out_path_folder = OUT_DIR / outputFolderName
    Path(out_path_folder).mkdir(parents=True, exist_ok=True)

    midi_files = getMidiByMetaData((MIDI_DATA_DIR).glob("**/*.midi"), genre_keywords=genre_keywords)


    for midi_path in tqdm(midi_files):
        tokens = tokenizer(midi_path)
        out_path = Path(out_path_folder, midi_path.stem + ".json")

        tokenizer.save_tokens(tokens, out_path)
        



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No command-line arguments provided for the tokenizer name.")
        sys.exit(1)
    tokenizeData(sys.argv[1] + '.json')
    

