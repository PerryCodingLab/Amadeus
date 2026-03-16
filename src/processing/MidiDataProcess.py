# needs modifications
from random import shuffle
from pathlib import Path
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI
import sys
import os
sys.path.append(os.path.abspath("./Discover-MIDI-Dataset"))
import json

# def labelMidi(midiPath):
#     with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
#         genre_data = json.load(f)
#     midiPath = Path(midiPath)
#     id_str = midiPath.parts[-3] + midiPath.parts[-2] + midiPath.stem
#     print(genre_data[0])
#     return genre_data[str(id_str)]
    

# Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
        genre_data = json.load(f)

count = 0
midi_paths_genre = {}
midi_paths = list()
for id, genre in genre_data.items():
    songpathstr = "./Discover-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id
    songpath = Path(songpathstr)
    midi_paths.append(songpath)
    midi_paths_genre[songpath] = genre

def get_genre_label(tok_seq, file_path):
    # Returns the integer ID corresponding to the loaded file
    return midi_paths_genre[file_path]

# midi_paths = list(Path("./Discover-MIDI-Dataset/MIDIs").glob("**/**/*.mid"))
total_num_files = len(midi_paths)

num_files_valid = round(total_num_files * 0.15)
num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

dataset  = DatasetMIDI(
    files_paths= midi_paths,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
    func_to_get_labels=get_genre_label
)




# # Chunk MIDIs and perform data augmentation on each subset independently
# for files_paths, subset_name in (
#     (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
# ):

#     # Split the MIDIs into chunks of sizes approximately about 1024 tokens
#     subset_chunks_dir = Path(f"dataset_{subset_name}")
#     split_files_for_training(
#         files_paths=files_paths,
#         tokenizer=tokenizer,
#         save_dir=subset_chunks_dir,
#         max_seq_len=1024,
#         num_overlap_bars=2,
#     )

#     # Perform data augmentation
#     augment_dataset(
#         subset_chunks_dir,
#         pitch_offsets=[-12, 12],
#         velocity_offsets=[-4, 4],
#         duration_offsets=[-0.5, 0.5],
#     )

