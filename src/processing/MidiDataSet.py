from MidiDataProcess import process_midi
from torch.utils.data import Dataset
from pathlib import Path
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI
from configs.config import MIDI_DATA_DIR
from symusic import Score
from typing import Iterable, Optional



def getMidiByMetaData(midi_root: Path,
    genre_keywords: Optional[Iterable[str]] = None,):
    midi_root = Path(midi_root)

    if genre_keywords is not None:
        genre_keywords = [g.lower() for g in genre_keywords]

    selected = []

    for midi_path in midi_root.rglob("**/*"):
        if not midi_path.is_file():
            continue
        if midi_path.suffix.lower() not in (".mid", ".midi"):
            continue
        if midi_path.stat().st_size < 14:
            continue

        try:
            meta_text = extract_midi_text_metadata(midi_path)
        except Exception:
            continue

        if genre_keywords is not None:
            if not any(g in meta_text for g in genre_keywords):
                continue

        selected.append(midi_path)

    return selected




def extract_midi_text_metadata(midi_path):
    """
    Extract all text-based metadata from a MIDI file.
    Returns a lowercase concatenated string.
    """
    score = Score(midi_path)

    texts = []

    for track in score.tracks:
        # Track name
        if track.name:
            texts.append(track.name)

        # Meta text events
        for event in track.events:
            if hasattr(event, "text") and event.text:
                texts.append(event.text)

    return " ".join(texts).lower()



midi_files = getMidiByMetaData(
    MIDI_DATA_DIR,
    genre_keywords=["jazz", "blues"]
)