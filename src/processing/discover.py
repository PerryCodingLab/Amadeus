import sys
import os
sys.path.append(os.path.abspath("./Discover-MIDI-Dataset/DATA/Genres_MIDIs"))
import json

with open("./Discover-MIDI-Dataset/DATA/Genres_MIDIs/genres_data.jsonl") as f:
    genre_data = json.load(f)

id_to_genre = {}
count = 0
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
count = 0
for genre in genre_data.items():
    print(genre[0])
    for band in genre[1].items():
        for singer in band[1].items():
            for SongID in singer[1]:
                # print(SongID)
                count +=1
                id_to_genre[str(SongID)] = genre[0]

print(count)
# with open('genre.jsonl', "w") as json_file:
#     json.dump(id_to_genre, json_file, indent= 4)