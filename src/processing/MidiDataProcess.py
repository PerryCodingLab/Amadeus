# needs modifications
from random import shuffle
from pathlib import Path
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI
from miditok import REMI
import sys
import os
import torch
sys.path.append(os.path.abspath("./Discover-MIDI-Dataset"))
sys.path.append(os.path.abspath("."))
from src.training.train_classifier import trainOnDataset
from models.GenreClassifier import MusicGenreClassifier
import json
from tqdm import tqdm 
# def labelMidi(midiPath):
#     with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
#         genre_data = json.load(f)
#     midiPath = Path(midiPath)
#     id_str = midiPath.parts[-3] + midiPath.parts[-2] + midiPath.stem
#     print(genre_data[0])
#     return genre_data[str(id_str)]

with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/index_to_genre.jsonl") as f:
    index_genre = json.load(f)

CONFIG = {
    'vocab_size': 300,        # Size of REMI vocabulary (adjust based on tokenization)
    'num_classes': 32,         # e.g., Rock, Jazz, Pop, Classical
    'd_model': 128,
    'batch_size': 32,
    'seq_len': 512,
    'epochs': 20,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def testModel():
    tokenizer = REMI()
    model = MusicGenreClassifier(vocab_size=CONFIG['vocab_size'], num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    model.load_state_dict(torch.load('266401.5376362best_classifier.pth', weights_only=False))
    model.eval()
    device = 'cuda'

    songNames = []
    prediction_genre = []

    midi_paths = list(Path("./Data/RawData/Random/").glob("*.mid"))
    for midi_file in midi_paths:
        try:
            tokens_seq = tokenizer.encode(midi_file)

            tokens = []
            for seq in tokens_seq:
                if len(seq.ids) > 0:
                    tokens.extend(seq.ids)
                    break
            tokens = tokens[:128]
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.to(device)
            with torch.no_grad():
                output = model(tokens)  # shape: [1, num_classes]
                # predicted_class_idx = output.argmax(dim=1).item()
            prediction = output.squeeze(0)

            # print("Single input shape:", tokens.shape)
            # print("Model output shape:", output.shape)
            # print("prediction shape:", prediction.shape)
            name = str(midi_file).split('\\')
            songNames.append(name[-1])
            prediction_genre.append(index_genre[str(prediction.argmax().item())])
        except Exception as e:
            print(f"Skipping {midi_file}: {e}")

    for i in range(len(songNames)):
        print("name: ", songNames[i])
        print("Prediction:", prediction_genre[i])
        print("\n")



if __name__ == "__main__":
    testModel()

    pass


# print("start pulling paths")
# # Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter

# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
#     genre_data = json.load(f)
# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre_to_index.jsonl") as f:
#     genre_index = json.load(f)



# midi_paths_genre = {}
# midi_paths = []
# count = 0
# for id, genre in tqdm(genre_data.items()):
#     count+=1
#     songpathstr = "./Discover-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id + ".mid"
#     songpath = Path(songpathstr)
#     midi_paths.append(songpath)
#     midi_paths_genre[songpath] = genre_index[genre]
#     if count >= 500:
#         break

# def get_genre_label(score ,tok_seq, file_path):
#     # Returns the integer ID corresponding to the loaded file
#     return midi_paths_genre[file_path]


# total_num_files = len(midi_paths)

# num_files_test = round(total_num_files * 0.15)
# shuffle(midi_paths)
# midi_paths_test = midi_paths[:num_files_test]
# midi_paths_train = midi_paths[num_files_test:]

# print("total number of midi files: ", total_num_files)
# tokenizer = REMI()


# dataset_test  = DatasetMIDI(
#     files_paths= midi_paths_test,
#     tokenizer=tokenizer,
#     max_seq_len=1024,
#     bos_token_id=tokenizer["BOS_None"],
#     eos_token_id=tokenizer["EOS_None"],
#     func_to_get_labels=get_genre_label
# )

# dataset_train  = DatasetMIDI(
#     files_paths= midi_paths_train,
#     tokenizer=tokenizer,
#     max_seq_len=1024,
#     bos_token_id=tokenizer["BOS_None"],
#     eos_token_id=tokenizer["EOS_None"],
#     func_to_get_labels=get_genre_label
# )

# print("start training")
# trainOnDataset(dataset_train, dataset_test, tokenizer)


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


