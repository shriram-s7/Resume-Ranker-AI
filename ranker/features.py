# ranker/features.py
"""
Stage 2: Feature extraction.
All three signal blocks computed here.
Every score is normalized to 0.0–1.0 before returning.
"""

import numpy as np
from datetime import date
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .constants import *
from .loader import build_semantic_text

# ── Load model once at module level ──────────────────────────────────────────
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(SBERT_MODEL)
    return _model

# ── JD embedding cached ───────────────────────────────────────────────────────
_jd_embedding = None

def get_jd_embedding() -> np.ndarray:
    global _jd_embedding
    if _jd_embedding is None:
        _jd_embedding = get_model().encode([JD_TEXT], convert_to_numpy=True)
    return _jd_embedding

# ─── BLOCK 1: SEMANTIC FIT ────────────────────────────────────────────────────

def compute_semantic_fit(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """
    Embed career narrative text for all candidates.
    Compare to JD embedding via cosine similarity.
    Returns array of shape (n,) with values 0.0–1.0.
    
    KEY DESIGN CHOICE: We embed career_history descriptions, NOT skills list.
    This catches Tier 5 candidates the JD describes: someone who built a
    recommendation system but never wrote 'RAG' in their profile.
    """
    model = get_model()
    jd_emb = get_jd_embedding()
    
    texts = [build_semantic_text(c) for c in candidates]
    candidate_embs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    
    scores = cosine_similarity(candidate_embs, jd_emb).flatten()
    
    # Normalize to 0–1 (cosine is already –1 to 1, shift and scale)
    scores = (scores + 1) / 2
    return scores.clip(0, 1)

# ─── BLOCK 2: STRUCTURAL FIT ──────────────────────────────────────────────────

def compute_structural_fit(candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_structural_score_0_to_1, breakdown_dict_for_explanation).
    
    Components:
    1. Years of experience proximity to ideal band
    2. Location match
    3. Title-chaser detection (average tenure across career)
    4. Pure consulting penalty
    5. Pure research disqualifier
    6. Notice period
    7. Education tier (mild)
    """
    p = candidate["profile"]
    s = candidate["redrob_signals"]
    history = candidate.get("career_history", [])
    breakdown = {}
    
    # 1. Years of experience
    yoe = p.get("years_of_experience", 0)
    if YOE_IDEAL_MIN <= yoe <= YOE_IDEAL_MAX:
        yoe_score = 1.0
    elif yoe < YOE_IDEAL_MIN:
        # Smooth falloff below ideal band — not a cliff
        yoe_score = max(0.1, yoe / YOE_IDEAL_MIN)
    else:
        # Overqualified — mild penalty
        yoe_score = max(0.6, 1 - (yoe - YOE_IDEAL_MAX) / 10)
    breakdown["years_of_experience"] = round(yoe_score, 3)
    
    location_str = (p.get("location", "") + " " + p.get("country", "")).lower()
    
    # Tier 1: JD's explicitly preferred cities
    TIER1 = ["pune", "noida"]
    # Tier 2: Other acceptable India metros  
    TIER2 = ["hyderabad", "delhi", "gurugram", "gurgaon", "mumbai", "bengaluru", "bangalore", "chennai", "ncr"]
    
    if any(loc in location_str for loc in TIER1):
        location_score = 1.0
    elif any(loc in location_str for loc in TIER2):
        location_score = 0.80
    elif "india" in location_str:
        location_score = 0.60   # India but tier-3 city
    elif s.get("willing_to_relocate", False):
        location_score = 0.30   # Overseas willing to relocate
    else:
        location_score = 0.10   # Overseas not willing — near-disqualifying
    breakdown["location"] = round(location_score, 3)
    
    # 3. Title-chaser detection
    # Average tenure < 14 months across 3+ roles = red flag per JD
    if len(history) >= 3:
        tenures = [h.get("duration_months", 0) for h in history if not h.get("is_current")]
        avg_tenure = np.mean(tenures) if tenures else 24
        if avg_tenure < 14:
            chaser_score = 0.3  # Strong penalty
        elif avg_tenure < 20:
            chaser_score = 0.65
        else:
            chaser_score = 1.0
    else:
        chaser_score = 0.8  # Not enough history to judge
    breakdown["tenure_stability"] = round(chaser_score, 3)
    
    # 4. Pure consulting penalty
    # If ENTIRE career is at consulting firms = penalty (not disqualification)
    company_names = [h.get("company", "").lower() for h in history]
    def is_consulting(company_name: str) -> bool:
        co = company_name.lower().strip()
        for firm in PURE_CONSULTING_FIRMS:
            if co == firm or co.startswith(firm + " ") or co.startswith(firm + ","):
                return True
        return False

    consulting_roles = sum(1 for co in company_names if is_consulting(co))
    if len(company_names) > 0 and consulting_roles == len(company_names):
        consulting_score = 0.3  # Pure consulting only career
    elif consulting_roles > 0:
        consulting_score = 0.75  # Some consulting but also product companies
    else:
        consulting_score = 1.0
    breakdown["company_type"] = round(consulting_score, 3)
    
    # 5. Pure research penalty
    all_desc = " ".join(h.get("description", "") for h in history).lower()
    research_hits = sum(1 for kw in RESEARCH_ONLY_KEYWORDS if kw in all_desc)
    production_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in all_desc)
    if research_hits >= 3 and production_hits == 0:
        research_score = 0.2  # Pure academic, no production
    elif research_hits > production_hits:
        research_score = 0.65
    else:
        research_score = 1.0
    breakdown["production_vs_research"] = round(research_score, 3)
    
    # 6. Notice period
    notice = s.get("notice_period_days")
    if notice is None:
        notice = 60  # unknown = neutral assumption, not ideal joiner
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.85
    else:
        notice_score = 0.70
    breakdown["notice_period"] = round(notice_score, 3)
    
    # 7. Education tier (mild signal, low weight)
    tier_map = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.70,
                "tier_4": 0.55, "unknown": 0.65}
    edu_list = candidate.get("education", [])
    if edu_list:
        best_tier = max(tier_map.get(e.get("tier", "unknown"), 0.65) for e in edu_list)
    else:
        best_tier = 0.60
    breakdown["education_tier"] = round(best_tier, 3)
    
    # Weighted combination of structural sub-scores
    structural_score = (
        0.25 * yoe_score +
        0.20 * location_score +
        0.20 * chaser_score +
        0.15 * consulting_score +
        0.10 * research_score +
        0.05 * notice_score +
        0.05 * best_tier
    )
    return round(structural_score, 4), breakdown

# ─── BLOCK 3: SKILLS FIT ──────────────────────────────────────────────────────

def compute_skills_fit(candidate: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Trust-weighted skill overlap with JD required + nice-to-have skills.
    
    KEY DESIGN CHOICE: We do NOT count skills by presence alone.
    Trust = f(endorsements, duration_months, proficiency level).
    A skill with 0 endorsements and 2 months duration = near zero weight.
    This is the direct countermeasure to keyword-stuffing.
    
    Also: if skill_assessment_score disagrees with claimed proficiency = penalty.
    """
    skills = candidate.get("skills", [])
    signals = candidate["redrob_signals"]
    assessment_scores = signals.get("skill_assessment_scores", {})
    
    breakdown = {}
    required_hit_scores = []
    nice_hit_scores = []
    
    for skill in skills:
        name = skill.get("name", "").lower()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration_months = skill.get("duration_months") or 6  # unknown duration = assume 6 months minimum, not zero
        
        # Base proficiency weight
        prof_weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.25)
        
        # Endorsement trust scale: 0 endorsements = 0.3x trust, 30+ = 1.0x
        endorse_trust = 0.3 + 0.7 * min(endorsements / ENDORSEMENT_TRUST_SCALE_MAX, 1.0)
        
        # Duration trust: 0 months = 0.2x, 24+ months = 1.0x
        duration_trust = 0.2 + 0.8 * min(duration_months / 24, 1.0)
        
        # Assessment cross-check penalty
        # If platform assessed this skill and score disagrees with claimed proficiency
        assess_multiplier = 1.0
        if name in {k.lower() for k in assessment_scores}:
            assessed_key = next(k for k in assessment_scores if k.lower() == name)
            assessed_val = assessment_scores[assessed_key] / 100  # normalize 0–1
            claimed_val = prof_weight
            disagreement = abs(assessed_val - claimed_val)
            if disagreement > 0.40:
                assess_multiplier = 0.6  # strong disagreement = lower trust
            elif disagreement > 0.20:
                assess_multiplier = 0.8
        
        trust_score = prof_weight * endorse_trust * duration_trust * assess_multiplier
        
        # Match against JD required skills
        for req_skill in JD_REQUIRED_SKILLS:
            if req_skill.lower() in name or name in req_skill.lower():
                required_hit_scores.append(trust_score)
                break
        
        # Match against nice-to-have skills
        for nice_skill in JD_NICE_TO_HAVE_SKILLS:
            if nice_skill.lower() in name or name in nice_skill.lower():
                nice_hit_scores.append(trust_score * 0.5)  # Half weight for nice-to-have
                break
    
    # Combine: sum of trust-weighted hits, normalized by JD skills count
    required_score = min(sum(required_hit_scores) / max(len(JD_REQUIRED_SKILLS) * 0.5, 1), 1.0)
    nice_score = min(sum(nice_hit_scores) / max(len(JD_NICE_TO_HAVE_SKILLS) * 0.3, 1), 1.0)
    
    # GitHub activity as an additional skill signal
    github = signals.get("github_activity_score", -1)
    if github == -1:
        github_bonus = 0.0  # No penalty for no GitHub — just no bonus
    else:
        github_bonus = (github / 100) * 0.15  # Up to 0.15 bonus
    
    total = min(0.70 * required_score + 0.30 * nice_score + github_bonus, 1.0)
    breakdown["required_skill_score"] = round(required_score, 3)
    breakdown["nice_to_have_score"] = round(nice_score, 3)
    breakdown["github_bonus"] = round(github_bonus, 3)
    
    return round(total, 4), breakdown

# ─── BEHAVIORAL MULTIPLIER ────────────────────────────────────────────────────

def compute_behavioral_multiplier(candidate: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Multiplicative gate on the base score. Range: BEHAVIORAL_FLOOR to 1.0.
    
    KEY DESIGN CHOICE: This is multiplicative, not additive.
    A dead profile cannot ride skill-score to top 10.
    But we floor at 0.3 — a technically perfect candidate who's gone quiet
    can still appear at rank 80, not be erased entirely.
    
    CONFIDENCE WEIGHTING on response rate:
    A 0.80 response_rate from someone with 2 profile views is noisy.
    A 0.80 from someone with 50 profile views is reliable.
    We scale confidence by profile_views_received_30d.
    """
    s = candidate["redrob_signals"]
    breakdown = {}
    today = date.today()
    
    # 1. Recency proxy — last_active_date doesn't exist in this dataset
    # Use profile_views_received_30d + search_appearance_30d as activity proxy
    views_30d = s.get("profile_views_received_30d", 0)
    search_30d = s.get("search_appearance_30d", 0)
    activity_signal = views_30d + search_30d
    # 400+ total = very active, 0 = completely dormant
    recency = max(0.40, min(1.0, 0.40 + (activity_signal / 500) * 0.60))
    breakdown["recency"] = round(recency, 3)
    
    # 2. Confidence-weighted response rate
    raw_response_rate = s.get("recruiter_response_rate", 0.5)
    
    # Confidence: high views + searches = this rate is based on real data
    signal_volume = min((views_30d + search_30d) / 100, 1.0)
    # If very low volume: regress response rate toward neutral prior (0.5)
    confident_response_rate = (
        signal_volume * raw_response_rate +
        (1 - signal_volume) * 0.5
    )
    # Never fully zero — silence isn't proof of disinterest
    response_factor = 0.4 + 0.6 * confident_response_rate
    breakdown["response_rate"] = round(confident_response_rate, 3)
    breakdown["response_factor"] = round(response_factor, 3)
    
    # 3. Interview completion rate
    completion = s.get("interview_completion_rate", 0.7)
    completion_factor = 0.5 + 0.5 * completion
    breakdown["interview_completion"] = round(completion, 3)
    
    # 4. Availability signals
    open_to_work = 1.0 if s.get("open_to_work_flag", False) else 0.85
    breakdown["open_to_work"] = open_to_work
    
    # 5. Market interest (other recruiters saving/searching this person = quality signal)
    saved_30d = s.get("saved_by_recruiters_30d", 0)
    market_signal = min(1.0 + saved_30d * 0.01, 1.15)  # Small bonus, capped at 1.15
    breakdown["saved_by_recruiters"] = saved_30d
    
    # Weighted combination
    raw_multiplier = (
        0.35 * recency +
        0.35 * response_factor +
        0.15 * completion_factor +
        0.10 * open_to_work +
        0.05 * min(market_signal, 1.0)
    )
    
    # Apply market signal as small boost on top
    multiplier = raw_multiplier * market_signal
    
    # Clamp to floor/ceil
    multiplier = max(BEHAVIORAL_FLOOR, min(BEHAVIORAL_CEIL, multiplier))
    breakdown["final_multiplier"] = round(multiplier, 3)
    
    return round(multiplier, 4), breakdown
