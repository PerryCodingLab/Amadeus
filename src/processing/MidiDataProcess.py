# needs modifications
from random import shuffle
from pathlib import Path
from miditok.data_augmentation import augment_dataset
from miditok.utils import split_files_for_training

# Split the dataset into train/valid/test subsets, with 15% of the data for each of the two latter
midi_paths = list(Path("path", "to", "dataset").glob("**/*.mid"))
total_num_files = len(midi_paths)
num_files_valid = round(total_num_files * 0.15)
num_files_test = round(total_num_files * 0.15)
shuffle(midi_paths)
midi_paths_valid = midi_paths[:num_files_valid]
midi_paths_test = midi_paths[num_files_valid:num_files_valid + num_files_test]
midi_paths_train = midi_paths[num_files_valid + num_files_test:]

# Chunk MIDIs and perform data augmentation on each subset independently
for files_paths, subset_name in (
    (midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")
):

    # Split the MIDIs into chunks of sizes approximately about 1024 tokens
    subset_chunks_dir = Path(f"dataset_{subset_name}")
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=1024,
        num_overlap_bars=2,
    )

    # Perform data augmentation
    augment_dataset(
        subset_chunks_dir,
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )