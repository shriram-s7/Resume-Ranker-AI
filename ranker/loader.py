# ranker/loader.py
import json
import gzip
from pathlib import Path
from typing import Iterator, List, Dict, Any

def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream candidates one at a time from .jsonl or .jsonl.gz.
    Memory-efficient — does not load all 100K into RAM at once during BM25 pass.
    """
    p = Path(path)
    if p.suffix == ".gz":
        opener = gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = open(p, "r", encoding="utf-8")
    
    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # skip malformed lines silently

def load_all_candidates(path: str) -> List[Dict[str, Any]]:
    """
    Load all candidates into memory. Used after BM25 filter reduces pool.
    At 5000 candidates post-filter, this is fine.
    """
    return list(stream_candidates(path))

def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Build a single searchable text string from all relevant fields of a candidate.
    Used for BM25 indexing.
    """
    p = candidate["profile"]
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    
    # Add career history descriptions
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    
    # Add skill names only (not proficiency — BM25 shouldn't trust keyword stuffing)
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    
    # Add certifications
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
    
    return " ".join(filter(None, parts)).lower()

def build_semantic_text(candidate: Dict[str, Any]) -> str:
    """
    Build text specifically for semantic embedding — focuses on career narrative,
    NOT skills list. This is where real signal lives per the JD.
    """
    parts = []
    p = candidate["profile"]
    parts.append(p.get("summary", ""))
    
    for role in candidate.get("career_history", []):
        # Weight recent roles more by repeating them
        desc = role.get("description", "")
        if role.get("is_current"):
            parts.append(desc)
            parts.append(desc)  # repeat current role for weight
        else:
            parts.append(desc)
    
    return " ".join(filter(None, parts))
