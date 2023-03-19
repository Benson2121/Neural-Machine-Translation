from math import exp
from typing import List, Sequence, Tuple
from a2_bleu_score import n_gram_precision, brevity_penalty


def METEOR_score(reference: Sequence[str], candidate: Sequence[str],
                 alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.3) -> float:
    """
    Calculate the METEOR score.

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    alpha: float, optional
        The weight assigned to precision in the METEOR score calculation
    beta: float, optional
        The weight assigned to recall in the METEOR score calculation
    gamma: float, optional
        The threshold value for F-mean calculation

    Returns
    -------
    meteor : float
        The METEOR score
    """
    # Calculate unigram, bigram, and trigram precision and recall
    p_n = [n_gram_precision(reference, candidate, i) for i in range(1, 4)]
    r_n = [n_gram_precision(candidate, reference, i) for i in range(1, 4)]

    # Calculate precision mean and recall mean
    p_mean = pow(pow(p_n[0], alpha) * pow(p_n[1], alpha) * pow(p_n[2], alpha), 1 / 3)
    r_mean = pow(pow(r_n[0], beta) * pow(r_n[1], beta) * pow(r_n[2], beta), 1 / 3)

    # Calculate F-mean
    F_mean = 0 if p_mean == 0 or r_mean == 0 else (1 - gamma) * (p_mean * r_mean) / (
            gamma * p_mean + (1 - gamma) * r_mean)

    # Calculate brevity penalty
    BP = brevity_penalty(reference, candidate)

    # Return the final METEOR score
    return F_mean * BP
