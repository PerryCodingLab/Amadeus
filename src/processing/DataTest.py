
import sys
import os
sys.path.append(os.path.abspath("./tegridy-tools/tegridy-tools"))
import TMIDIX


def pathtoscore(input_midi):
    fdata = open(input_midi, 'rb').read()
    raw_score = TMIDIX.midi2score(fdata)
    # print(raw_score)
    escore = TMIDIX.advanced_score_processor(raw_score, 
                                             return_enhanced_score_notes=True, 
                                            )[0]
    first_note_index = [e[0] for e in raw_score[1]].index('note')
    
    meta_data = raw_score[1][:first_note_index] + [escore[0]] + [escore[-1]] + [raw_score[1][-1]]
    print('Input MIDI metadata:', meta_data[:5])


root = "./Tegridy-MIDI-Dataset/Tegridy-MIDI-Dataset-CC-BY-NC-SA"
count = 0

for file in os.listdir(root):
    if file.endswith(".mid"):
        path = os.path.join(root, file)
        pathtoscore(path)
        count+=1
        if count > 1:
            break

    