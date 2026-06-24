from datetime import date
from typing import Dict, Any, List

def generate_reasoning(
    candidate: Dict[str, Any],
    scored: Dict[str, Any],
    rank: int,
    fragility_score: float = None
) -> str:
    from ranker.constants import JD_REQUIRED_SKILLS, JD_NICE_TO_HAVE_SKILLS

    p = candidate["profile"]
    s = candidate["redrob_signals"]
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "unknown title")
    career = candidate.get("career_history", [])
    current_roles = [h for h in career if h.get("is_current")]
    company = (
        current_roles[0].get("company") if current_roles
        else (career[0].get("company") if career else "unknown company")
    )
    location = p.get("location", "unknown location")

    # Skills that appear in almost every candidate — skip these in display
    GENERIC_SKILLS = {
        "python", "machine learning", "nlp", "ranking", "retrieval",
        "vector database", "embeddings", "transformer", "fine-tuning",
        "deep learning", "data science", "statistics"
    }

    all_jd = [sk.lower() for sk in JD_REQUIRED_SKILLS + JD_NICE_TO_HAVE_SKILLS]

    def skill_trust(sk):
        return sk.get("endorsements", 0) * sk.get("duration_months", 0)

    # Priority 1: specific named tools — Pinecone, Qdrant, FAISS, LangChain, etc.
    tool_skills = []
    for sk in skills:
        name = sk.get("name", "").lower()
        is_jd_match = any(jd in name or name in jd for jd in all_jd)
        is_generic = name in GENERIC_SKILLS
        if is_jd_match and not is_generic:
            tool_skills.append(sk)
    tool_skills_sorted = sorted(tool_skills, key=skill_trust, reverse=True)

    # Priority 2: any JD skill including generic
    any_jd_skills = []
    for sk in skills:
        name = sk.get("name", "").lower()
        if any(jd in name or name in jd for jd in all_jd):
            any_jd_skills.append(sk)
    any_jd_sorted = sorted(any_jd_skills, key=skill_trust, reverse=True)

    # Build final skill list — pick 2 unique names
    chosen = []
    seen_names = set()
    for sk in tool_skills_sorted:
        n = sk.get("name", "").lower()
        if n not in seen_names:
            chosen.append(sk)
            seen_names.add(n)
        if len(chosen) >= 2:
            break

    if len(chosen) < 2:
        for sk in any_jd_sorted:
            n = sk.get("name", "").lower()
            if n not in seen_names:
                chosen.append(sk)
                seen_names.add(n)
            if len(chosen) >= 2:
                break

    if len(chosen) < 2:
        for sk in sorted(skills, key=skill_trust, reverse=True):
            n = sk.get("name", "").lower()
            if n not in seen_names:
                chosen.append(sk)
                seen_names.add(n)
            if len(chosen) >= 2:
                break

    skill_names = [sk["name"] for sk in chosen]

    # Behavioral signals
    response_rate = s.get("recruiter_response_rate", 0)
    views_30d = s.get("profile_views_received_30d", 0)
    search_30d = s.get("search_appearance_30d", 0)
    activity_total = views_30d + search_30d
    notice = s.get("notice_period_days", 60)

    sem = scored["semantic_score"]
    struct = scored["structural_score"]
    sk_score = scored["skills_score"]
    mult = scored["behavioral_multiplier"]

    if rank <= 30:
        skill_str = ", ".join(skill_names) if skill_names else "relevant AI/ML skills"

        recent_role = history[0] if history else {}
        recent_desc = recent_role.get("description", "")
        production_hint = ""
        if any(kw in recent_desc.lower() for kw in ["deployed", "production", "users", "scaled", "shipped"]):
            production_hint = " Career narrative confirms production deployment experience."

        if activity_total >= 200 and response_rate >= 0.6:
            availability = f"Highly active ({activity_total} recruiter interactions/30d); {int(response_rate*100)}% response rate."
        elif activity_total >= 50:
            availability = f"Moderately active ({activity_total} interactions/30d); {int(response_rate*100)}% response rate."
        else:
            availability = f"Low platform activity ({activity_total} interactions/30d); response rate {int(response_rate*100)}%."

        concern = ""
        struct_bd = scored.get("structural_breakdown", {})
        if struct_bd.get("tenure_stability", 1.0) < 0.5:
            concern = " Concern: short average tenure across roles."
        elif struct_bd.get("company_type", 1.0) < 0.5:
            concern = " Note: career predominantly at services firms."
        elif notice > 60:
            concern = f" Notice period of {notice}d may require negotiation."

        frag_note = ""
        if fragility_score is not None:
            if fragility_score < 0.05:
                frag_note = " Fit is robust — score holds even without top skill."
            elif fragility_score > 0.15:
                frag_note = " Fit somewhat concentrated on top skill."

        sentence1 = f"{yoe:.1f}-year {title} at {company} ({location}); strong match on {skill_str}.{production_hint}"
        sentence2 = f"{availability}{concern}{frag_note}"

    elif rank <= 70:
        skill_str = ", ".join(skill_names) if skill_names else "partial skill overlap"
        sentence1 = (
            f"{yoe:.1f} years as {title} at {company}; "
            f"partial JD match on {skill_str} (semantic: {sem:.2f}, skills: {sk_score:.2f})."
        )
        if mult < 0.55:
            sentence2 = f"Behavioral signals weak ({int(response_rate*100)}% response rate) — ranked lower despite technical fit."
        else:
            sentence2 = f"Behavioral signals acceptable ({int(response_rate*100)}% response rate); technical fit is limiting factor."

    else:
        skill_str = skill_names[0] if skill_names else "limited overlap"
        sentence1 = f"{yoe:.1f}-year {title}; limited JD alignment (semantic: {sem:.2f}, skills: {sk_score:.2f})."
        struct_issue = " Structural fit is the main gap." if struct < 0.4 else ""
        sentence2 = f"Ranked {rank} based on partial signals.{struct_issue}"

    return f"{sentence1} {sentence2}".strip()
