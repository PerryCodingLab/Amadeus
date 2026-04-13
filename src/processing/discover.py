import sys
import os
sys.path.append(os.path.abspath("./Discover-MIDI-Dataset/DATA/Genres_MIDIs"))
import json
from pathlib import Path
from miditok import REMI
from tqdm import tqdm 
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
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

def saveTokens(preprocessed_tokens,preprocessed_labels):
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_tokens, preprocessed_labels, test_size=0.2, random_state=42
    )
    # stratify=preprocessed_labels
    torch.save(X_train, "Data/ProccessedData/train_tokens.pt")
    torch.save(y_train, "Data/ProccessedData/train_labels.pt")
    torch.save(X_test, "Data/ProccessedData/test_tokens.pt")
    torch.save(y_test, "Data/ProccessedData/test_labels.pt")


if __name__ == "__main__":
    tokenizer = REMI()
    BroadMap = subGenreCount()
    # X1,Y1 = tokenizeDiscover(BroadMap, tokenizer)
    X2,Y2 = tokenizeGodzilla(BroadMap, tokenizer)
    # X1.extend(X2)
    # Y1.extend(Y2)
    # saveTokens(X1,Y1)


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