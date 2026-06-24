#!/usr/bin/env python3
"""
rank.py — Resume Ranker AI Hackathon Submission Entry Point

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (enforced):
    - CPU only
    - No network calls
    - ≤5 minutes wall-clock
    - ≤16GB RAM
    - Output: exactly 100 rows, ranks 1–100, non-increasing scores
"""

import argparse
import csv
import time
from pathlib import Path
from tqdm import tqdm

from ranker.loader import load_all_candidates, stream_candidates, build_candidate_text
from ranker.retrieval import build_bm25_index, retrieve_top_k
from ranker.features import compute_semantic_fit
from ranker.scorer import score_candidates, compute_counterfactual_fragility
from ranker.diversity import mmr_rerank
from ranker.reasoning import generate_reasoning


def parse_args():
    p = argparse.ArgumentParser(description="Resume Ranker AI candidate ranker")
    p.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    p.add_argument("--out", default="submission.csv", help="Output CSV path")
    p.add_argument("--top-k-bm25", type=int, default=5000, help="BM25 retrieval pool size")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    
    print("=" * 60)
    print("RESUME RANKER AI")
    print("=" * 60)
    
    # ── Stage 1: Load all candidates ──────────────────────────────────────
    print(f"\n[1/6] Loading candidates from {args.candidates}...")
    all_candidates = load_all_candidates(args.candidates)
    print(f"      Loaded {len(all_candidates):,} candidates in {time.time()-t0:.1f}s")
    
    # ── Stage 2: BM25 retrieval filter ───────────────────────────────────
    print(f"\n[2/6] Building BM25 index and retrieving top-{args.top_k_bm25}...")
    t1 = time.time()
    bm25 = build_bm25_index(all_candidates)
    top_k_results = retrieve_top_k(all_candidates, bm25, k=args.top_k_bm25)
    filtered_candidates = [all_candidates[idx] for idx, _ in top_k_results]
    print(f"      Retrieved {len(filtered_candidates):,} candidates in {time.time()-t1:.1f}s")
    
    # ── Stage 3: Semantic embedding ───────────────────────────────────────
    print(f"\n[3/6] Computing semantic fit (SBERT embeddings)...")
    t2 = time.time()
    semantic_scores = compute_semantic_fit(filtered_candidates)
    print(f"      Embedded {len(filtered_candidates):,} candidates in {time.time()-t2:.1f}s")
    
    # ── Stage 4: Full scoring ─────────────────────────────────────────────
    print(f"\n[4/6] Scoring candidates...")
    t3 = time.time()
    scored_candidates = score_candidates(filtered_candidates, semantic_scores)
    print(f"      Scored {len(scored_candidates):,} candidates in {time.time()-t3:.1f}s")
    
    # ── Stage 5: MMR Reranking ───────────────────────────────────────────
    print(f"\n[5/6] Performing MMR diversity re-ranking...")
    t4 = time.time()
    top_scored_candidates = mmr_rerank(scored_candidates, top_n=100)
    print(f"      Selected top 100 candidates in {time.time()-t4:.1f}s")
    
    # Sort by final score descending and candidate ID ascending to ensure strict compliance
    top_scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # Create candidate map for easy lookup
    candidate_map = {c["candidate_id"]: c for c in filtered_candidates}
    
    # ── Stage 6: Reasoning and Output Generation ────────────────────────
    print(f"\n[6/6] Generating reasoning and saving results...")
    t5 = time.time()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        prev_score = float("inf")
        for rank, scored in enumerate(top_scored_candidates, 1):
            cid = scored["candidate_id"]
            cand = candidate_map[cid]
            
            fragility = compute_counterfactual_fragility(cand, scored)
            scored["fragility_score"] = fragility
            
            reasoning = generate_reasoning(cand, scored, rank, fragility)
            
            # Enforce strictly non-increasing scores
            final_score = min(scored["final_score"], prev_score)
            prev_score = final_score
            
            writer.writerow([cid, rank, f"{final_score:.6f}", reasoning])
            
    print(f"      Saved 100 ranked candidates to {args.out} in {time.time()-t5:.1f}s")
    print(f"Total time elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
