"""
myllm.evaluation.metrics — Pure metric calculation utilities.
Zero dependencies on model or intelligence layers to prevent circular imports.
"""

from typing import List, Set


def analyze_repetition(text: str) -> float:
    """
    Calculate a repetition score (0.0 to 1.0) based on repeating 3-grams.
    Returns 0.0 for diverse non-repetitive text, higher for degenerate loops.
    """
    words = text.split()
    if len(words) < 3:
        return 0.0

    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0

    unique_trigrams = set(trigrams)
    repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
    return repetition_ratio
