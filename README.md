# Resume Ranker AI — Intelligent Candidate Discovery & Ranking System

This repository contains a professional-grade, competition-ready candidate ranking system built for the Redrob Hackathon challenge: "Intelligent Candidate Discovery & Ranking."

The system (Resume Ranker AI) is optimized to discover, score, and rank the top 100 candidates for a specific **Senior AI Engineer (Founding Team at Redrob AI)** job description from a database of up to 100,000 profiles. It executes strictly on CPU in under 5 minutes, completely offline with zero external network API calls during ranking.

---

## ⚡ Key Pipeline Stages

The discovery pipeline is structured into a 6-stage architecture:

1. **Stage 1: BM25 Retrieval Filter (`retrieval.py`)**
   * Performs sub-second sparse text search across the entire pool of 100k+ candidates.
   * Filters candidate list down to the top 5,000 matches before running expensive neural embeddings.
2. **Stage 2: Semantic Career Narrative Embedding (`features.py`)**
   * Uses the cached `all-MiniLM-L6-v2` Sentence-Transformer model (22MB, CPU-friendly) to embed candidate career descriptions.
   * Compares them against the Job Description via cosine similarity. 
   * Weights recent and current roles higher. Does not trust the skills keyword list to prevent keyword-stuffing hacks.
3. **Stage 3: Trust-Weighted Skill Overlap (`features.py`)**
   * Matches candidate skills against required and nice-to-have skills.
   * Trust weight is calculated as a function of skill endorsements, duration, and claimed proficiency.
   * Incorporates recruiter assessment cross-check penalties and GitHub activity scores.
4. **Stage 4: Structural & Career Stability Fit (`features.py`)**
   * Scores experience proximity to the ideal band (5–9 years).
   * Verifies location matches (prefers Indian hubs: Pune, Noida, NCR, Hyderabad, etc.).
   * Flags "title-chasers" (average tenure < 14 months).
   * Penalizes candidates with careers exclusively at service-based consulting firms.
   * Penalizes pure-academic research profiles that lack production deployment keywords.
5. **Stage 5: Honeypots and Title Denylists (`traps.py`)**
   * Disqualifies candidates with titles outside the role family (Marketing, HR, Finance, etc.).
   * Detects and blocks fabricated honeypot profiles using duration inconsistencies, salary inversions, and expert skills with 0 usage.
6. **Stage 6: MMR Diversity Re-ranking & Fragility Check (`diversity.py`, `scorer.py`, `reasoning.py`)**
   * Applies Maximal Marginal Relevance (MMR) re-ranking (Lambda=0.8) to prevent identical candidate profiles from crowding the top 100.
   * Computes a counterfactual fragility score (performance drop when the best skill is ablated) for top candidates.
   * Generates rank-consistent, fact-grounded natural language reasoning for the matches.

---

## 📁 Repository Structure

```
resume-ranker-ai/
├── rank.py                    # Main CLI entry point
├── app.py                     # Streamlit UI dashboard
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── submission_metadata.yaml   # metadata file
├── validate_submission.py     # validation utility script
└── ranker/
    ├── __init__.py
    ├── loader.py              # streaming loader and text builders
    ├── retrieval.py           # BM25 sparse filter
    ├── features.py            # SBERT semantic fit & structural logic
    ├── scorer.py              # score fusion and fragility checks
    ├── traps.py               # honeypot filter and title denylist
    ├── diversity.py           # MMR diversity reranking
    ├── reasoning.py           # rank-consistent reasoning generator
    └── constants.py           # weights, thresholds, denylists, and JD text
```

---

## ⚙️ Installation & Setup

Ensure Python is installed on your system. Install the project dependencies using the provided `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

---

## 🧪 Testing the Pipeline

### 1. Generate Mock Candidate Data
To test the pipeline on realistic candidate structures, run our mock data generator script:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/shriram-s7/Resume-Ranker-AI/main/scratch/generate_mock_data.py', 'generate_mock_data.py')"
# Or execute the script already located in your local scratch folder:
python -m scratch.generate_mock_data
```
This saves 1,000 mock profiles containing normal matches, consulting-only, research-only, title-chasers, and honeypot anomalies to `candidates.jsonl`.

### 2. Execute CLI Ranker
Run the main pipeline command:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
This reads candidates, executes the pipeline, and saves the top 100 candidates to `submission.csv`.

### 3. Verify Submission Compliance
Validate that the resulting CSV passes all competition formatting and sorting rules:

```bash
python validate_submission.py submission.csv
```

---

## 📊 Streamlit UI Dashboard

To explore candidate rankings and inspect individual candidate details, launch the interactive UI:

```bash
streamlit run app.py
```

### UI Features:
* **Run Pipeline**: Run discovery on your local candidate dataset with real-time progress bars.
* **Analytics**: Interactive dashboards showing score distribution, years of experience, locations, and top candidate titles.
* **Candidate Inspector**: Drill down into any top 100 candidate to see detailed sub-scores (semantic, structural, skills), notice period details, career stability logs, and system reasoning.
