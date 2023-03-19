'''
Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x
from typing import List, Sequence, Iterable


def grouper(seq:Sequence[str], n:int) -> List:
    """Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    ngrams = []
    for i in range(len(seq) - n + 1):
        ngrams.append(tuple(seq[i:i+n]))
    return ngrams


def n_gram_precision(reference:Sequence[str], candidate:Sequence[str], n:int) -> float:
    """Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    count_i = 0
    re_grouper = grouper(reference, n)
    ngrams = grouper(candidate, n)
    for ngram in ngrams:
        if ngram in re_grouper:
            count_i += 1
    return count_i / len(ngrams) if len(ngrams) != 0 else 0


def brevity_penalty(reference:Sequence[str], candidate:Sequence[str]) -> float:
    """Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    if len(candidate) == 0:
        return 0
    c = len(candidate)
    r = len(reference)
    if r < c:
        return 1
    else:
        return exp(1 - r / c)


def BLEU_score(reference:Sequence[str], candidate:Sequence[str], n) -> float:
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    p_n = [n_gram_precision(reference, candidate, i) for i in range(1, n+1)]
    p_order = 1
    for i in range(len(p_n)):
        p_order *= p_n[i]
    BP = brevity_penalty(reference, candidate)
    return BP * (p_order ** (1 / n))
