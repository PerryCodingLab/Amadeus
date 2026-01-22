import numpy as np
from typing import List, Dict

class ContentPreservationEvaluator:
    """
    Evaluates if the harmonic content of a song is preserved after transformation.
    
    Applicable Models:
    - Genre Shift Model (Adversarial Autoencoder)
    """

    def _get_pitch_histogram(self, midi_pitches: List[int]) -> np.ndarray:
        """
        Converts a sequence of MIDI pitches into a normalized 12-bin Pitch Class Histogram.
        """
        if not midi_pitches:
            return np.zeros(12)
            
        # 1. Initialize 12 bins (C, C#, D, ..., B)
        histogram = np.zeros(12)
        
        # 2. Populate bins (Modulo 12 removes octave info)
        for pitch in midi_pitches:
            pitch_class = pitch % 12
            histogram[pitch_class] += 1
            
        # 3. Normalize (so sum equals 1.0)
        total_notes = len(midi_pitches)
        if total_notes > 0:
            histogram = histogram / total_notes
            
        return histogram

    def calculate_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Math helper to calculate Cosine Similarity between two vectors.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    def evaluate_harmonic_preservation(self, input_seq: List[int], output_seq: List[int]) -> Dict:
        """
        Compares the harmonic profile of the Input (Source Genre) vs Output (Target Genre).
        
        Args:
            input_seq: List of MIDI pitches from the original file.
            output_seq: List of MIDI pitches from the generated file.
            
        Returns:
            Dictionary containing:
            - 'similarity_score': Float (0.0 to 1.0).
            - 'input_hist': The normalized histogram of the input.
            - 'output_hist': The normalized histogram of the output.
        """
        hist_input = self._get_pitch_histogram(input_seq)
        hist_output = self._get_pitch_histogram(output_seq)
        
        score = self.calculate_cosine_similarity(hist_input, hist_output)
        
        return {
            "similarity_score": round(score, 4),
            "input_hist": hist_input, # Useful for debugging/plotting
            "output_hist": hist_output
        }

# --- Example Usage ---
if __name__ == "__main__":
    evaluator = ContentPreservationEvaluator()
    
    # 1. Original Jazz Lick (C Major-ish)
    # Notes: C, E, G, B (C Maj 7)
    input_jazz = [60, 64, 67, 71, 60, 64] 
    
    # 2. Rock Transformation (Output)
    # The model adds a bass line (low C) and repeats notes, but keeps the Key.
    # Notes: C2, C4, E4, G4, C4 (Still C Major)
    output_rock = [36, 60, 64, 67, 60]
    
    # 3. Bad Transformation (Output)
    # The model hallucinated and changed the key to F# (F#, A#, C#)
    output_bad = [66, 70, 73, 66]

    result_good = evaluator.evaluate_harmonic_preservation(input_jazz, output_rock)
    print(f"Good Transformation Score: {result_good['similarity_score']}")
    # Expect High Score (~0.9) because both are C-Major heavy.

    result_bad = evaluator.evaluate_harmonic_preservation(input_jazz, output_bad)
    print(f"Bad Transformation Score: {result_bad['similarity_score']}")
    # Expect Low Score (~0.0) because C Major and F# Major share almost no notes.