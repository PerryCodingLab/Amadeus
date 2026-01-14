import numpy as np
from typing import List, Tuple, Dict

class MusicalCoherenceEvaluator:
    """
    A rule-based evaluator to measure the musical validity of generated symbolic music.
    
    Applicable Models:
    - Extension Model (RNN/LSTM)
    - Jamming Partner
    """

    def __init__(self):
        # 1. Define Scale Patterns (Intervals relative to the root note)
        # Major scale intervals: W-W-H-W-W-W-H (0, 2, 4, 5, 7, 9, 11)
        self.major_intervals = {0, 2, 4, 5, 7, 9, 11}
        # Minor scale intervals (Natural Minor): W-H-W-W-H-W-W (0, 2, 3, 5, 7, 8, 10)
        self.minor_intervals = {0, 2, 3, 5, 7, 8, 10}
        
        # Pre-compute all 12 Major and 12 Minor scales
        self.scales = self._generate_all_scales()

    def _generate_all_scales(self) -> Dict[str, set]:
        """
        Generates pitch classes (0-11) for all 12 major and minor keys.
        """
        scales = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i, root_name in enumerate(note_names):
            # Generate Major Scale for this root
            major_notes = {(i + interval) % 12 for interval in self.major_intervals}
            scales[f"{root_name} Major"] = major_notes
            
            # Generate Minor Scale for this root
            minor_notes = {(i + interval) % 12 for interval in self.minor_intervals}
            scales[f"{root_name} Minor"] = minor_notes
            
        return scales

    def check_scale_consistency(self, midi_pitches: List[int]) -> Dict:
        """
        Determines the most likely key of the sequence and calculates how many
        notes fit that key.
        
        Args:
            midi_pitches: A list of MIDI pitch integers (0-127).
            
        Returns:
            Dictionary containing:
            - 'best_key': The name of the key that fits best (e.g., 'C Major').
            - 'score': 0.0 to 1.0 (percentage of notes that fit the key).
            - 'in_key_notes': Count of notes strictly in key.
            - 'total_notes': Total notes processed.
        """
        if not midi_pitches:
            return {"best_key": None, "score": 0.0}

        # Convert MIDI pitches (0-127) to Pitch Classes (0-11)
        # e.g., 60 (C4) becomes 0, 61 (C#4) becomes 1
        pitch_classes = [p % 12 for p in midi_pitches]
        unique_pitch_classes = set(pitch_classes)
        
        best_key = None
        best_match_count = -1
        
        # Check against all pre-computed scales
        for key_name, scale_notes in self.scales.items():
            # intersection: notes in our sequence that are ALSO in this scale
            match_count = 0
            for pc in pitch_classes:
                if pc in scale_notes:
                    match_count += 1
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_key = key_name
        
        score = best_match_count / len(pitch_classes) if len(pitch_classes) > 0 else 0
        
        return {
            "best_key": best_key,
            "score": round(score, 4), # e.g., 0.9523
            "in_key_notes": best_match_count,
            "total_notes": len(pitch_classes)
        }

    def check_rhythmic_stability(self, durations: List[int], grid_resolution: int = 120) -> Dict:
        """
        Checks if the generated durations align with a standard rhythmic grid.
        
        Args:
            durations: List of duration values (in ticks or token IDs depending on encoding).
            grid_resolution: The base tick value for a beat or sub-beat (e.g., 120 ticks).
                             If using token IDs, this logic might need adjustment to check
                             against a set of 'valid' duration tokens instead.
            
        Returns:
            Dictionary containing:
            - 'on_grid_ratio': Percentage of notes landing perfectly on the grid.
            - 'is_stable': Boolean, true if ratio > threshold (e.g. 0.8).
        """
        if not durations:
            return {"on_grid_ratio": 0.0}
            
        on_grid_count = 0
        for d in durations:
            # Check if duration is a multiple of the grid (or close to it)
            # This assumes 'd' is in ticks. If 'd' is a class index, we map it first.
            if d > 0 and (d % grid_resolution == 0):
                on_grid_count += 1
                
        ratio = on_grid_count / len(durations)
        
        return {
            "on_grid_ratio": round(ratio, 4),
            "is_stable": ratio > 0.85 # Threshold for stability
        }

# --- Example Usage ---
if __name__ == "__main__":
    evaluator = MusicalCoherenceEvaluator()
    
    # 1. Simulate a "Good" C Major melody (C, D, E, G, C)
    good_melody = [60, 62, 64, 67, 72] 
    print(f"Good Melody Check: {evaluator.check_scale_consistency(good_melody)}")
    
    # 2. Simulate a "Bad" Random melody (Chromatic mess)
    # 60=C, 61=C#, 62=D, 63=D# (Very unlikely to be in one key)
    bad_melody = [60, 61, 62, 63, 66, 69]
    print(f"Bad Melody Check: {evaluator.check_scale_consistency(bad_melody)}")