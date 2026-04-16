import sys
import os
sys.path.append(os.path.abspath("./Discover-MIDI-Dataset/DATA/Genres_MIDIs"))
sys.path.append(os.path.abspath("."))
import json
from pathlib import Path
from miditok import REMI
from tqdm import tqdm 
from src.utils import suppress_c_stdout
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from miditok import TokenizerConfig
from miditok.utils import split_files_for_training

# from src.utils import suppress_c_stdout
# import gc
# gc.collect()
# torch.cuda.empty_cache()
# print(torch.version.cuda)
# print(torch.cuda.is_available())  # should be True
# print(torch.cuda.device_count())  # number of GPUs
# print(torch.cuda.get_device_name(0))  # GPU name

def tokenizeDiscover(broadMap, tokenizer):
    max_seq_len = 1024  # same as training
    PAD_TOKEN = tokenizer["PAD_None"]

    with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
        genre_data = json.load(f)
    with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre_to_index.jsonl") as f:
        genre_index = json.load(f)

    preprocessed_tokens = []
    preprocessed_labels = []
    midi_paths_genre = {}
    midi_paths = []

    # count = 0
    for id, genre in tqdm(genre_data.items()):
        # count+=1
        songpathstr = "./Discover-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id + ".mid"
        songpath = Path(songpathstr)
        try:
            tokens_seq = tokenizer.encode(songpath)
            tokens = []
            for seq in tokens_seq:
                if len(seq.ids) > 0:
                    tokens.extend(seq.ids)
                else: break

                # Pad or truncate to max_seq_len
            if len(tokens) < max_seq_len:
                tokens += [PAD_TOKEN] * (max_seq_len - len(tokens))
            else:
                tokens = tokens[:max_seq_len]

            # tokens = tokens[0].ids
            preprocessed_tokens.append(tokens)
            preprocessed_labels.append(genre_index[broadMap[genre]])  # 0..num_classes-1
            midi_paths_genre[songpath] = genre_index[genre]
        except Exception as e:
            print(f"Skipping {songpath}: {e}")
        # if count >= 5000:
        #     break
    return preprocessed_tokens, preprocessed_labels
    

def subGenreCount():
    with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/Subgenre_to_index.jsonl") as f:
        genre_index = json.load(f)
    with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/subgenre_to_genre.jsonl") as f:
        BroadMap = json.load(f)
    Broad = {}
    for sub, index in genre_index.items():
        B = BroadMap[sub]
        if B not in Broad:
            Broad[B] = []
        Broad[B].append(sub)

    reverse_dict = {}

    for key, value in Broad.items():
        for sub in value:
            reverse_dict[sub] = key

    return reverse_dict
    # counts = {genre: len(subgenres) for genre, subgenres in Broad.items()}

    # for genre, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{genre}: {count}")
def tokenizeGodzilla(broadMap, tokenizer):
    max_seq_len = 1024  # same as training
    PAD_TOKEN = tokenizer["PAD_None"]

    with open("./Godzilla-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
        genre_data = json.load(f)
    with open("./Godzilla-MIDI-Dataset/DATA/Genres_MIDIs/genre_to_index.jsonl") as f:
        genre_index = json.load(f)

    preprocessed_tokens = []
    preprocessed_labels = []
    midi_paths_genre = {}
    midi_paths = []

    count = 0
    for id, genre in tqdm(genre_data.items()):
        count+=1
        songpathstr = "./Godzilla-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id + ".mid"
        songpath = Path(songpathstr)
        try:
            tokens_seq = tokenizer.encode(songpath)
            tokens = []
            for seq in tokens_seq:
                if len(seq.ids) > 0:
                    tokens.extend(seq.ids)
                else: break

                # Pad or truncate to max_seq_len
            if len(tokens) < max_seq_len:
                tokens += [PAD_TOKEN] * (max_seq_len - len(tokens))
            else:
                tokens = tokens[:max_seq_len]

            # tokens = tokens[0].ids
            preprocessed_tokens.append(tokens)
            preprocessed_labels.append(genre_index[broadMap[genre]])  # 0..num_classes-1
            midi_paths_genre[songpath] = genre_index[genre]
        except Exception as e:
            print(f"Skipping {songpath}: {e}")
        if count >= 100:
            break

def saveTokens(preprocessed_tokens,preprocessed_labels, name):
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_tokens, preprocessed_labels, test_size=0.2, random_state=42
    )
    # stratify=preprocessed_labels
    torch.save(X_train, f"Data/ProccessedData/train_tokens{name}.pt")
    torch.save(y_train, f"Data/ProccessedData/train_labels.pt{name}")
    torch.save(X_test, f"Data/ProccessedData/test_tokens.pt{name}")
    torch.save(y_test, f"Data/ProccessedData/test_labels.pt{name}")


def saveTokensNoLabels(preprocessed_tokens, name):
    torch.save(preprocessed_tokens, f"Data/ProccessedData/{name}_train_tokens.pt")
    # X_train, X_test = train_test_split(
    #     preprocessed_tokens, test_size=0.2, random_state=42
    # )

    # torch.save(X_train, f"Data/ProccessedData/{name}_train_tokens.pt")
    # torch.save(X_test, f"Data/ProccessedData/{name}_test_tokens.pt")





def tokenize_and_save_locally(midi_folder_path, output_vocab_path, name):
    """Runs locally to convert MIDI to integers and save to disk."""
    # 1. Initialize tokenizer
    config = TokenizerConfig(num_velocities=16, use_chords=True, use_rests=True)
    tokenizer = REMI(config)
    all_midi_paths = []
    def stream_midi_files(folders):
        for folder in folders:
            # Path.glob() yields files one by one dynamically
            for path in Path(folder).resolve().glob('**/*.mid'):
                yield path
    # for i in midi_folder_path:
    #     midi_paths = list(Path(i).resolve().glob('**/*.mid'))
    #     all_midi_paths.extend(midi_paths)

        # print("Batch size: ", len(midi_paths))
        # print([str(s) for s in midi_paths])
    print(f"Total files found: {len(all_midi_paths)}")
    
    def chunk_tokens(tokens, max_len=1024):
        """Slices a flat list of tokens into chunks of max_len."""
        return [tokens[i : i + max_len] for i in range(0, len(tokens), max_len)]
    
    batch_size = 1000
    current_batch_sequences = []
    processed_log = "processed_midi_log.txt"
    # 1. Load the list of already processed files so we can resume
    processed_files = set()
    if os.path.exists(processed_log):
        with open(processed_log, 'r', encoding='utf-8') as f:
            processed_files = set(f.read().splitlines())

    print(f"Skipping {len(processed_files)} previously processed files...")

    batch_index = len(processed_files) // batch_size
    files_in_current_batch = 0

    # print(f"Found {len(midi_paths)} MIDI files. Tokenizing...")


    # Create a new list to hold the successful file paths in RAM
    current_batch_paths = []

    # 2. Tokenize
    for path in stream_midi_files(midi_folder_path):
        path_str = str(path)
        if path_str in processed_files:
            continue
        try:
            tokenized_midi = tokenizer(path)
            if isinstance(tokenized_midi, list): 
                for track in tokenized_midi:
                    if len(track.ids) > 0:
                        current_batch_sequences.extend(chunk_tokens(track.ids))
            else:
                if len(tokenized_midi.ids) > 0:
                    current_batch_sequences.extend(chunk_tokens(tokenized_midi.ids))
            
            current_batch_paths.append(path_str)
            files_in_current_batch += 1

            #saving a batch
            if files_in_current_batch >= batch_size:
                batch_name = f"{name}/part_{batch_index}"
                saveTokensNoLabels(current_batch_sequences, batch_name)
                print(f"Saved {batch_name}!")
                with open(processed_log, 'a', encoding='utf-8') as log_file:

                    log_file.write('\n'.join(current_batch_paths) + '\n')
                # Reset the list to free up RAM for the next batch
                current_batch_paths = []
                current_batch_sequences = []
                files_in_current_batch = 0
                batch_index += 1
        except Exception as e:
            print(f"Could not parse {path}: {e}")


    if len(current_batch_sequences) > 0:
        batch_name = f"{name}/part_{batch_index}"
        saveTokensNoLabels(current_batch_sequences, batch_name)
        print(f"Saved final batch: {batch_name}!")
        if current_batch_paths: # Make sure the list isn't empty
            with open(processed_log, 'a', encoding='utf-8') as log_file:
                log_file.write('\n'.join(current_batch_paths) + '\n')
 
    # 3. Export the sequences to JSON
    # with open(output_data_path, 'w') as f:
    #     json.dump(all_token_sequences, f)
        
    # 4. Export the tokenizer configuration
    tokenizer.save(output_vocab_path)
    
    print(f"✅ Saved {len(current_batch_sequences)} sequences")
    print(f"✅ Saved tokenizer params to {output_vocab_path}")


if __name__ == "__main__":
    # choice = int(input("if tokenizing with genre labels press 0, for prediction press 2: "))
    choice = 2
    if choice == 0:
        print("tokenizing genre")
        tokenizer = REMI()
        BroadMap = subGenreCount()
        # X1,Y1 = tokenizeDiscover(BroadMap, tokenizer)
        X2,Y2 = tokenizeGodzilla(BroadMap, tokenizer)
        # X1.extend(X2)
        # Y1.extend(Y2)
        # saveTokens(X1,Y1)
    elif choice == 2:

        # midi_folders = ["./Data/RawData/A/A", "./Data/RawData/A/B"]
        # midi_folders = ["./Data/RawData/MIDIS"]
        allsets = ["./Godzilla-MIDI-Dataset/MIDIs", "./Discover-MIDI-Dataset/MIDIs", "./lmd_matched"]
        # folderName = input("whats the name of the folder? ")
        folderName = "Predict"
        Path(f"Data/ProccessedData/{folderName}").mkdir(parents=True, exist_ok=True)
        tokenize_and_save_locally(allsets, "./Data/tokenizers", folderName)
        print("tokenizing prediction")
        
        # subsets = []
        # # subsets.extend(["./Godzilla-MIDI-Dataset/MIDIs/" + str(i) for i in range(10)])
        # # subsets.extend(["./Godzilla-MIDI-Dataset/MIDIs/" + str(i) for i in ['a', 'b', 'c', 'd', 'e', 'f']])
        # subsets.extend(["./Discover-MIDI-Dataset/MIDIs/" + str(i) for i in range(10)])
        # subsets.extend(["./Discover-MIDI-Dataset/MIDIs/" + str(i) for i in ['a', 'b', 'c', 'd', 'e', 'f']])
        # subsubsets = []

        # alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        # subsubsets.extend(["./lmd_matched/" +  letter for letter in alphabet])

        # for sub in subsets:
        #     subsubsets.extend([str(sub) + "/" + str(i) for i in range(10)])
        #     subsubsets.extend([str(sub) + "/" + str(i) for i in ['a', 'b', 'c', 'd', 'e', 'f']])
        # folderName = input("whats the name of the folder? ")

        # for midi_folders in subsubsets:
        #     print(midi_folders)
        #     Path(f"Data/ProccessedData/{folderName}").mkdir(parents=True, exist_ok=True)
        #     tokenize_and_save_locally(midi_folders, "./Data/tokenizers", folderName)
        #     print("tokenizing prediction")
    elif choice == 3:
        data = torch.load("Data/ProccessedData/test_train_tokens.pt")
        for i in data:
            print(len(i))
    
# subset_chunks_dir = Path("./Data/ProccessedData/split").resolve()
#     split_files_for_training(
#         files_paths=midi_paths,
#         tokenizer=tokenizer,
#         save_dir=subset_chunks_dir,
#         max_seq_len=1024,
#         num_overlap_bars=2,
#     )

# torch.save(preprocessed_tokens, "Data/ProccessedData/tokenized_midi.pt")
# torch.save(preprocessed_labels, "Data/ProccessedData/labels.pt")

# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
#     genre_data = json.load(f)
# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre_to_index.jsonl") as f:
#     genre_index = json.load(f)

# preprocessed_tokens = []
# preprocessed_labels = []
# tokenizer = REMI()
# for id, genre in genre_data.items():
#     songpathstr = "./Discover-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id + ".mid"
#     songpath = Path(songpathstr)
#     tokens = tokenizer.midi_to_tokens(songpath)
#     preprocessed_tokens.append(tokens)
#     preprocessed_labels.append(genre_index[genre]) 


# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genres_data.jsonl") as f:
#     genre_data = json.load(f)
# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre_to_index.jsonl") as f:
#     genre_index = json.load(f)
# with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genre.jsonl") as f:
#     genre_data = json.load(f)

# id_to_genre = {}
# count = 0

# midi_paths_genre = {}
# midi_paths = []
# for id, genre in genre_data.items():
#     count+=1
#     songpathstr = "./Discover-MIDI-Dataset/MIDIs/" + id[0] +"/"+ id[1]+"/" + id + ".mid"
#     songpath = Path(songpathstr)
#     midi_paths.append(songpath)
#     midi_paths_genre[songpath] = genre
#     if count >= 5:
#         break

# print(genre_index.items())
# for songid, genre in genre_data.items():
#     count+=1
#     midi_paths_genre[songid] = genre_index[genre]
#     if count >= 5:
#         break
# print(midi_paths_genre)




# for genre, band in genre_data.items():
#     count +=1
#     if count == 5:
#         break
#     print(genre)
#     # print(singer)
#     for band, song in band.items():
#         # id_to_genre[str(song_id)] = genre
#         for singerID, song_id in song.items():
#             print(song_id)



# count = 0
# for genre in genre_data.items():
#     print(genre[0])
#     for band in genre[1].items():
#         for singer in band[1].items():
#             for SongID in singer[1]:
#                 # print(SongID)
#                 count +=1
#                 id_to_genre[str(SongID)] = genre[0]
#                 if count >=2:
#                     break
    
# index_to_genre = {}
# for genre in enumerate(genre_data.items()):
#     # print(genre[1][0])
#     index_to_genre[genre[0]] = genre[1][0]

    

# print(len(index_to_genre))
# with open('index_to_genre.jsonl', "w") as json_file:
#     json.dump(index_to_genre, json_file, indent= 4)