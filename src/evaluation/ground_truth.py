import difflib
from typing import List, Dict

class GroundTruthEvaluator:
    """
    Evaluates generated music against the actual original continuation (Ground Truth).
    
    Applicable Models:
    - Extension Model (RNN/LSTM)
    """

    def calculate_edit_distance(self, generated_seq: List[int], target_seq: List[int]) -> int:
        """
        Calculates the Levenshtein distance between two sequences.
        This is the minimum number of edits needed to turn 'generated' into 'target'.
        """
        size_x = len(generated_seq) + 1
        size_y = len(target_seq) + 1
        
        # Create a matrix of zeros
        matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]
        
        # Initialize first row and column
        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y
            
        # Fill the matrix
        for x in range(1, size_x):
            for y in range(1, size_y):
                if generated_seq[x-1] == target_seq[y-1]:
                    matrix[x][y] = matrix[x-1][y-1]
                else:
                    matrix[x][y] = min(
                        matrix[x-1][y] + 1,     # Deletion
                        matrix[x][y-1] + 1,     # Insertion
                        matrix[x-1][y-1] + 1    # Substitution
                    )
                    
        return matrix[size_x-1][size_y-1]

    def calculate_similarity_ratio(self, generated_seq: List[int], target_seq: List[int]) -> float:
        """
        Calculates a normalized similarity score between 0.0 and 1.0 
        using Python's built-in SequenceMatcher (similar to BLEU logic).
        
        Returns:
            Float: 1.0 is a perfect match, 0.0 is no similarity.
        """
        if not generated_seq and not target_seq:
            return 1.0
            
        matcher = difflib.SequenceMatcher(None, generated_seq, target_seq)
        return matcher.ratio()

# --- Example Usage ---
if __name__ == "__main__":
    evaluator = GroundTruthEvaluator()
    
    # Example: Model tries to predict "C, D, E"
    real_target = [60, 62, 64]
    
    # Scenario 1: Perfect prediction
    gen_perfect = [60, 62, 64]
    print(f"Perfect - Edit Dist: {evaluator.calculate_edit_distance(gen_perfect, real_target)}")
    print(f"Perfect - Similarity: {evaluator.calculate_similarity_ratio(gen_perfect, real_target)}")

    # Scenario 2: Model missed one note (Output: C, D, F)
    gen_mistake = [60, 62, 65]
    print(f"Mistake - Edit Dist: {evaluator.calculate_edit_distance(gen_mistake, real_target)}")
    print(f"Mistake - Similarity: {evaluator.calculate_similarity_ratio(gen_mistake, real_target)}")