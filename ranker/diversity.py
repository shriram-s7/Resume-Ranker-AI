import numpy as np
from typing import List, Dict, Any
from .constants import MMR_LAMBDA, MMR_CANDIDATE_POOL

def build_feature_vector(scored: Dict[str, Any]) -> np.ndarray:
    return np.array([
        scored["semantic_score"],
        scored["structural_score"],
        scored["skills_score"],
        scored.get("behavioral_multiplier", 0.7),
    ])

def mmr_rerank(
    scored_candidates: List[Dict[str, Any]],
    top_n: int = 100
) -> List[Dict[str, Any]]:
    # Sort by score, remove disqualified
    pool = sorted(scored_candidates, key=lambda x: -x["final_score"])
    pool = [p for p in pool if not p["is_honeypot"] and not p["is_disqualified"]]
    pool = pool[:MMR_CANDIDATE_POOL]

    # Dedup by candidate_id first (strict)
    seen_ids = set()
    id_deduped = []
    for p in pool:
        if p["candidate_id"] not in seen_ids:
            seen_ids.add(p["candidate_id"])
            id_deduped.append(p)
    pool = id_deduped

    # Score-fingerprint dedup at 4dp — catches truly identical scored profiles
    seen_fp = {}
    fp_deduped = []
    for p in pool:
        fp = (
            round(p["semantic_score"], 4),
            round(p["structural_score"], 4),
            round(p["skills_score"], 4),
        )
        if fp not in seen_fp:
            seen_fp[fp] = p["candidate_id"]
            fp_deduped.append(p)
    pool = fp_deduped

    if len(pool) <= top_n:
        return pool[:top_n]

    feature_vectors = np.array([build_feature_vector(p) for p in pool])
    scores = np.array([p["final_score"] for p in pool])

    score_range = scores.max() - scores.min()
    norm_scores = (scores - scores.min()) / score_range if score_range > 0 else np.ones_like(scores)

    selected_indices = []
    remaining = list(range(len(pool)))

    for _ in range(top_n):
        if not remaining:
            break
        if not selected_indices:
            best = max(remaining, key=lambda i: norm_scores[i])
        else:
            selected_vecs = feature_vectors[selected_indices]
            def mmr_score(i):
                relevance = norm_scores[i]
                norms = np.linalg.norm(selected_vecs, axis=1) * np.linalg.norm(feature_vectors[i]) + 1e-8
                sims = np.dot(selected_vecs, feature_vectors[i]) / norms
                return MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * sims.max()
            best = max(remaining, key=mmr_score)
        selected_indices.append(best)
        remaining.remove(best)

    return [pool[i] for i in selected_indices]
