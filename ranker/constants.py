# ranker/constants.py
# All tunable parameters in one place. Change here, affects everything.

# ─── Job Description ────────────────────────────────────────────────────────

JD_TEXT = """
Senior AI Engineer role at Redrob AI. Requires production experience with 
embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI embeddings),
vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, 
OpenSearch), strong Python, evaluation frameworks for ranking systems (NDCG, MRR, MAP, 
A/B testing, offline benchmarks). Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), 
learning-to-rank (XGBoost), HR-tech or marketplace product experience, distributed systems, 
open-source AI contributions. Role involves owning the intelligence layer: ranking, retrieval, 
matching systems. First 90 days: audit BM25 rule-based system, ship v2 ranking with embeddings 
and hybrid retrieval, build evaluation infrastructure. Needs product-company experience, 
scrappy shipping mindset, 5-9 years experience, Pune or Noida India location preferred.
"""

JD_REQUIRED_SKILLS = [
    "embeddings", "sentence-transformers", "vector database", "FAISS", "Pinecone",
    "Weaviate", "Qdrant", "Elasticsearch", "OpenSearch", "hybrid search",
    "retrieval", "ranking", "NDCG", "MRR", "A/B testing", "Python",
    "recommendation systems", "information retrieval", "semantic search",
    "fine-tuning", "LLM", "transformer", "NLP", "machine learning"
]

JD_NICE_TO_HAVE_SKILLS = [
    "LoRA", "QLoRA", "PEFT", "XGBoost", "learning to rank", "distributed systems",
    "open source", "LangChain", "RAG", "Hugging Face", "PyTorch", "TensorFlow"
]

# ─── Scoring Weights ─────────────────────────────────────────────────────────

WEIGHT_SEMANTIC   = 0.40   # Career description vs JD embedding similarity
WEIGHT_STRUCTURAL = 0.30   # Years, location, trajectory, company type
WEIGHT_SKILLS     = 0.30   # Trust-weighted skill overlap

# Behavioral multiplier range (never zeros out a candidate completely)
BEHAVIORAL_FLOOR  = 0.30
BEHAVIORAL_CEIL   = 1.00

# ─── Experience Band ─────────────────────────────────────────────────────────

YOE_IDEAL_MIN  = 5.0
YOE_IDEAL_MAX  = 9.0
YOE_HARD_MIN   = 2.0   # Below this, very unlikely — soft penalty only
YOE_HARD_MAX   = 15.0  # Above this, overqualified concern

# ─── Location ────────────────────────────────────────────────────────────────

PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "delhi", "gurugram", "gurgaon",
    "mumbai", "bengaluru", "bangalore", "chennai", "ncr"
]

# ─── Title Denylist (hard gate — score = 0 regardless of skills) ─────────────

DISQUALIFYING_TITLE_KEYWORDS = [
    "marketing", "hr ", "human resource", "content writer", "copywriter",
    "finance", "accountant", "sales", "graphic design", "ui designer",
    "ux designer", "product designer", "recruiter", "talent acquisition",
    "business development", "operations manager", "supply chain",
    "customer support", "customer success", "social media",
    "product manager"
]

# ─── Consulting Firm Denylist ─────────────────────────────────────────────────
# Penalize candidates whose ENTIRE career is at these firms (not just one stint)

PURE_CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "mphasis", "hexaware",
    "mindtree", "l&t infotech", "ltimindtree", "hcl technologies", "hcl tech"
]

# ─── Research-only Disqualifier Keywords ─────────────────────────────────────
# Career descriptions full of these with NO production keywords = penalize

RESEARCH_ONLY_KEYWORDS = ["arxiv", "published", "lab", "phd", "laboratory", "thesis", "conference paper"]
PRODUCTION_KEYWORDS    = ["deployed", "production", "users", "latency", "a/b test", "api", "shipped", "scaled"]

# ─── Honeypot Detection ───────────────────────────────────────────────────────

HONEYPOT_MAX_EXPERT_SKILLS_ZERO_DURATION = 3  # More than this = flag
HONEYPOT_MIN_SUSPICIOUS_SKILL_COUNT      = 15  # More skills than plausible
HONEYPOT_SALARY_INVERSION                = True  # min > max in expected salary

# ─── MMR Diversity ────────────────────────────────────────────────────────────

MMR_LAMBDA         = 0.80   # 0.8 = mostly score-driven, 0.2 = diversity push
MMR_CANDIDATE_POOL = 150    # Score top 150, then MMR-select top 100

# ─── BM25 Retrieval ──────────────────────────────────────────────────────────

BM25_TOP_K         = 5000   # Retrieve top 5000 before full scoring
BM25_MIN_SCORE     = 0.01   # Discard complete non-matches

# ─── Sentence Transformer Model ──────────────────────────────────────────────

SBERT_MODEL        = "all-MiniLM-L6-v2"  # 22MB, fast CPU inference

# ─── Notice Period ────────────────────────────────────────────────────────────

NOTICE_IDEAL_DAYS  = 30
NOTICE_OK_DAYS     = 60
NOTICE_PENALTY_MULTIPLIER = 0.90  # Applied if notice > 60 days

# ─── Skill Proficiency Weights ────────────────────────────────────────────────

PROFICIENCY_WEIGHTS = {
    "beginner":     0.25,
    "intermediate": 0.55,
    "advanced":     0.80,
    "expert":       1.00
}

# ─── Endorsement Trust Scaling ────────────────────────────────────────────────
# A skill with 0 endorsements is not trusted at face value

ENDORSEMENT_TRUST_SCALE_MAX = 30  # Endorsements above this = full trust
