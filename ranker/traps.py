# ranker/traps.py
"""
Stage 4: Honeypot and trap detection.
These checks run AFTER scoring but BEFORE final ranking.
Flagged candidates are hard-excluded from the top 100.

Per the spec: ~80 honeypot candidates with subtly impossible profiles.
Submissions with >10% honeypot rate in top 100 are disqualified.
"""

from datetime import date
from typing import Dict, Any, Tuple
from .constants import *

def check_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Returns (is_honeypot, reason_string).
    
    Honeypot signals (per submission_spec.md):
    1. Experience at a company founded AFTER their stated start date
       (we can't check founding dates, but we CAN check duration impossibilities)
    2. 'expert' proficiency in many skills with 0 duration_months
    3. Total skill count implausibly high for years_of_experience
    4. Career history total duration >> years_of_experience (fabricated timeline)
    5. Salary range where min > max (data inconsistency signal)
    """
    p = candidate["profile"]
    s = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    
    # Check 1: Expert skills with zero usage duration
    expert_zero_duration = [
        sk for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 0) == 0
    ]
    if len(expert_zero_duration) >= HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION:
        return True, f"Claims expert in {len(expert_zero_duration)} skills with 0 months usage"
    
    # Check 2: Implausibly many skills for experience level
    yoe = p.get("years_of_experience", 0)
    plausible_skill_max = max(5, int(yoe * 3))  # ~3 new skills per year is generous
    if len(skills) > HONEYPOT_MIN_SUSPICIOUS_SKILL_COUNT and len(skills) > plausible_skill_max:
        return True, f"{len(skills)} skills claimed but only {yoe} years experience"
    
    # Check 3: Total career duration vs stated YOE
    total_career_months = sum(h.get("duration_months", 0) for h in history)
    stated_months = yoe * 12
    if total_career_months > stated_months * 1.6 and total_career_months > 24:
        return True, f"Career history totals {total_career_months}mo but YOE implies {int(stated_months)}mo"
    
    # Check 4: Salary inversion (min > max is a data fabrication signal)
    salary = s.get("expected_salary_range_inr_lpa", {})
    if salary.get("min", 0) > salary.get("max", 999):
        # NOTE: This alone doesn't disqualify — 26% of real candidates have this
        # (likely a data generation artifact). Only flag if combined with other signals.
        other_flags = len(expert_zero_duration) > 0
        if other_flags:
            return True, "Multiple consistency failures including salary inversion"
    
    # Check 5: All skills at maximum proficiency (statistically impossible)
    if len(skills) >= 8:
        expert_count = sum(1 for sk in skills if sk.get("proficiency") == "expert")
        if expert_count / len(skills) > 0.7:
            return True, f"{expert_count}/{len(skills)} skills claimed as 'expert'"
    
    return False, ""

def is_title_disqualified(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Hard title gate — marketing, HR, content, finance, product managers etc."""
    title = candidate["profile"].get("current_title", "").lower().strip()
    for kw in DISQUALIFYING_TITLE_KEYWORDS:
        if kw.strip() in title:
            return True, f"Title '{candidate['profile']['current_title']}' is outside target role family"
    return False, ""
