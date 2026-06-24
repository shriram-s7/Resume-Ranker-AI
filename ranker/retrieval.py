# ranker/retrieval.py
"""
Stage 1: BM25 retrieval filter.
Reduces 100K candidates to ~5000 before expensive scoring.
Sub-second on CPU. No model weights needed.
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
from .constants import JD_TEXT, BM25_TOP_K
from .loader import build_candidate_text
import numpy as np

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r'\b\w+\b', text.lower())

def build_bm25_index(candidates: List[Dict[str, Any]]) -> BM25Okapi:
    """Build BM25 index over all candidate texts."""
    corpus = [tokenize(build_candidate_text(c)) for c in candidates]
    return BM25Okapi(corpus)

def retrieve_top_k(
    candidates: List[Dict[str, Any]],
    bm25: BM25Okapi,
    k: int = BM25_TOP_K
) -> List[Tuple[int, float]]:
    """
    Returns list of (original_index, bm25_score) for top-k candidates.
    """
    query_tokens = tokenize(JD_TEXT)
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k indices by score
    top_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((int(idx), float(scores[idx])))
    
    return results
