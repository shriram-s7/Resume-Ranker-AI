# ranker/scorer.py
"""
Stage 3: Weighted fusion scoring.
Combines all three blocks + behavioral multiplier into final score.
Also runs counterfactual fragility check on top candidates.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .features import (
    compute_semantic_fit, compute_structural_fit,
    compute_skills_fit, compute_behavioral_multiplier
)
from .traps import check_honeypot, is_title_disqualified
from .constants import *
from tqdm import tqdm

def score_candidates(
    candidates: List[Dict[str, Any]],
    semantic_scores: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Full scoring pipeline for a pre-filtered candidate list.
    semantic_scores: precomputed array from compute_semantic_fit()
    
    Returns list of scored candidate dicts with all breakdown fields.
    """
    results = []
    
    for i, candidate in enumerate(tqdm(candidates, desc="Scoring candidates")):
        cid = candidate["candidate_id"]
        
        # ── Hard gates first (cheapest checks) ────────────────────────────
        is_honeypot, hp_reason = check_honeypot(candidate)
        is_disqualified, dq_reason = is_title_disqualified(candidate)
        
        if is_honeypot or is_disqualified:
            results.append({
                "candidate_id": cid,
                "final_score": 0.0,
                "base_score": 0.0,
                "behavioral_multiplier": 0.0,
                "semantic_score": round(float(semantic_scores[i]), 4),
                "structural_score": 0.0,
                "skills_score": 0.0,
                "structural_breakdown": {},
                "skills_breakdown": {},
                "behavioral_breakdown": {},
                "is_honeypot": is_honeypot,
                "is_disqualified": is_disqualified,
                "disqualify_reason": hp_reason or dq_reason,
                "fragility_score": None,
            })
            continue
        
        # ── Feature extraction ─────────────────────────────────────────────
        semantic_score = float(semantic_scores[i])
        structural_score, struct_breakdown = compute_structural_fit(candidate)
        skills_score, skills_breakdown = compute_skills_fit(candidate)
        behavioral_mult, behavioral_breakdown = compute_behavioral_multiplier(candidate)
        
        # ── Base score (additive) ──────────────────────────────────────────
        base_score = (
            WEIGHT_SEMANTIC   * semantic_score +
            WEIGHT_STRUCTURAL * structural_score +
            WEIGHT_SKILLS     * skills_score
        )
        
        # ── Final score (multiplicative behavioral gate) ───────────────────
        final_score = base_score * behavioral_mult
        
        results.append({
            "candidate_id": cid,
            "final_score": round(final_score, 6),
            "base_score": round(base_score, 6),
            "behavioral_multiplier": round(behavioral_mult, 4),
            "semantic_score": round(semantic_score, 4),
            "structural_score": round(structural_score, 4),
            "skills_score": round(skills_score, 4),
            "structural_breakdown": struct_breakdown,
            "skills_breakdown": skills_breakdown,
            "behavioral_breakdown": behavioral_breakdown,
            "is_honeypot": False,
            "is_disqualified": False,
            "disqualify_reason": "",
            "fragility_score": None,
        })
    
    return results

def compute_counterfactual_fragility(
    candidate: Dict[str, Any],
    scored_result: Dict[str, Any]
) -> float:
    """
    Score drop when best skill is removed.
    High drop = fragile = riding one keyword.
    Low drop = robust = broadly qualified.
    
    Only run this on top ~50 to save compute.
    """
    from copy import deepcopy
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0
    
    # Find highest-trust skill
    best_skill_idx = max(range(len(skills)),
        key=lambda i: skills[i].get("endorsements", 0) * skills[i].get("duration_months", 0))
    
    # Create candidate without best skill
    ablated = deepcopy(candidate)
    ablated["skills"].pop(best_skill_idx)
    
    ablated_skills_score, _ = compute_skills_fit(ablated)
    original_skills_score = scored_result["skills_score"]
    
    drop = original_skills_score - ablated_skills_score
    return round(drop, 4)
