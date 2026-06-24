import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from pathlib import Path

from ranker.constants import JD_TEXT, JD_REQUIRED_SKILLS, JD_NICE_TO_HAVE_SKILLS
from ranker.loader import load_all_candidates, build_candidate_text
from ranker.retrieval import build_bm25_index, retrieve_top_k
from ranker.features import compute_semantic_fit
from ranker.scorer import score_candidates, compute_counterfactual_fragility
from ranker.diversity import mmr_rerank
from ranker.reasoning import generate_reasoning

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Ranker AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .candidate-card {
        border-left: 5px solid #FF4B4B;
        background-color: #fafafa;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='text-align: center; color: #FF4B4B; margin-top: 0;'>🎯 Resume Ranker AI</h2>", unsafe_allow_html=True)
st.sidebar.title("Configuration")
candidates_file_path = st.sidebar.text_input("Local Candidates Path", value="candidates.jsonl")
top_k_bm25 = st.sidebar.number_input("BM25 Retrieval Pool Size", min_value=100, max_value=10000, value=5000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Load Precomputed")
st.sidebar.write("To avoid the 5-minute SBERT embedding run on CPU for the full 100K candidates, load the precomputed CSV instantly.")
precomputed_csv_path = st.sidebar.text_input("Precomputed CSV Path", value="submission.csv")
load_precomputed_btn = st.sidebar.button("Load Precomputed CSV")

if load_precomputed_btn:
    t_start = time.time()
    csv_path = Path(precomputed_csv_path)
    jsonl_path = Path(candidates_file_path)
    
    if not csv_path.exists():
        st.sidebar.error(f"CSV file `{precomputed_csv_path}` not found.")
    elif not jsonl_path.exists():
        st.sidebar.error(f"Candidates file `{candidates_file_path}` not found.")
    else:
        with st.spinner("Loading and scoring precomputed candidates..."):
            try:
                # 1. Read CSV
                df_pre = pd.read_csv(csv_path)
                target_ids = set(df_pre["candidate_id"].tolist())
                reasoning_map = dict(zip(df_pre["candidate_id"], df_pre["reasoning"]))
                rank_map = dict(zip(df_pre["candidate_id"], df_pre["rank"]))
                score_map = dict(zip(df_pre["candidate_id"], df_pre["score"]))
                
                # 2. Fast look up of 100 candidate profiles in JSONL
                matched_profiles = {}
                if jsonl_path.suffix == ".gz":
                    import gzip
                    opener = gzip.open(jsonl_path, "rt", encoding="utf-8")
                else:
                    opener = open(jsonl_path, "r", encoding="utf-8")
                
                with opener as f:
                    for line in f:
                        idx = line.find('"candidate_id": "')
                        if idx != -1:
                            start = idx + len('"candidate_id": "')
                            end = line.find('"', start)
                            if end != -1:
                                cid = line[start:end]
                                if cid in target_ids:
                                    matched_profiles[cid] = json.loads(line)
                                    if len(matched_profiles) == len(target_ids):
                                        break
                
                # 3. Score just the 100 loaded candidates
                ordered_candidates = [matched_profiles[cid] for cid in df_pre["candidate_id"] if cid in matched_profiles]
                
                if not ordered_candidates:
                    st.sidebar.error("No matching candidate profiles found in JSONL path.")
                else:
                    semantic_scores = compute_semantic_fit(ordered_candidates)
                    scored_candidates = score_candidates(ordered_candidates, semantic_scores)
                    
                    candidate_map = {c["candidate_id"]: c for c in ordered_candidates}
                    scored_map = {s["candidate_id"]: s for s in scored_candidates}
                    
                    display_rows = []
                    for rank, cid in enumerate(df_pre["candidate_id"], 1):
                        if cid not in candidate_map or cid not in scored_map:
                            continue
                        cand = candidate_map[cid]
                        scored = scored_map[cid]
                        fragility = compute_counterfactual_fragility(cand, scored)
                        
                        display_rows.append({
                            "Rank": rank,
                            "Candidate ID": cid,
                            "Score": round(float(score_map[cid]), 4),
                            "Years Exp": cand["profile"]["years_of_experience"],
                            "Location": cand["profile"]["location"],
                            "Title": cand["profile"]["current_title"],
                            "Company": next((h.get("company","Unknown") for h in cand.get("career_history",[]) if h.get("is_current")), cand.get("career_history",[{}])[0].get("company","Unknown")),
                            "Reasoning": reasoning_map[cid],
                            "scored_data": scored,
                            "raw_candidate": cand,
                            "fragility_score": fragility
                        })
                    
                    st.session_state.pipeline_results = {
                        "display_df": pd.DataFrame(display_rows),
                        "raw_results": display_rows,
                        "metrics": {
                            "total_candidates": 100000 if "original" in candidates_file_path else 1000,
                            "filtered_candidates": len(ordered_candidates),
                            "time_load": 0.0,
                            "time_ret": 0.0,
                            "time_sem": 0.0,
                            "time_score": 0.0,
                            "time_mmr": 0.0,
                            "total_time": time.time() - t_start
                        }
                    }
                    st.sidebar.success("Successfully loaded precomputed results!")
            except Exception as e:
                st.sidebar.error(f"Error loading precomputed results: {str(e)}")


# ── Main Content Header ────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Resume Ranker AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hackathon Challenge — End-to-End CPU Ranking System</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_jd, tab_pipeline, tab_analytics, tab_inspector = st.tabs([
    "💼 Job Description", 
    "⚡ Run Pipeline", 
    "📈 Analytics Dashboard", 
    "🔍 Candidate Inspector"
])

# ── Tab 1: Job Description ───────────────────────────────────────────────────
with tab_jd:
    st.subheader("Target Job Description")
    st.info("Role: Founding Senior AI Engineer (Resume Ranker AI)")
    st.text_area("Job Description", value=JD_TEXT.strip(), height=250, disabled=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Required Skills")
        st.write(", ".join([f"`{sk}`" for sk in JD_REQUIRED_SKILLS]))
    with col2:
        st.write("### Nice-To-Have Skills")
        st.write(", ".join([f"`{sk}`" for sk in JD_NICE_TO_HAVE_SKILLS]))

# ── Session State for Pipeline Results ────────────────────────────────────────
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# ── Tab 2: Run Pipeline ───────────────────────────────────────────────────────
with tab_pipeline:
    st.subheader("Execute Candidate Discovery & Ranking")
    st.write("Run the pipeline on your candidate dataset. This streams candidates, runs the BM25 retrieval filter, embeds narratives, scores structural & skill parameters, filters honeypots, performs MMR reranking, and outputs the top 100.")
    
    # Check candidates file existence
    file_exists = Path(candidates_file_path).exists()
    if file_exists:
        st.success(f"Found file: `{candidates_file_path}`")
    else:
        st.error(f"Candidates file `{candidates_file_path}` not found. Please create it or verify the path.")
        
    run_btn = st.button("Run Pipeline", disabled=not file_exists)
    
    if run_btn:
        progress_text = st.empty()
        bar = st.progress(0)
        
        # 1. Loading
        progress_text.text("Stage 1: Streaming candidates and indexing...")
        bar.progress(15)
        t_start = time.time()
        all_candidates = load_all_candidates(candidates_file_path)
        t_load = time.time() - t_start
        
        # 2. BM25 Retrieval Filter
        progress_text.text(f"Stage 2: Building BM25 index and filtering (Top-{top_k_bm25})...")
        bar.progress(35)
        t_ret_start = time.time()
        bm25 = build_bm25_index(all_candidates)
        top_k_results = retrieve_top_k(all_candidates, bm25, k=top_k_bm25)
        filtered_candidates = [all_candidates[idx] for idx, _ in top_k_results]
        t_ret = time.time() - t_ret_start
        
        # 3. Semantic similarity
        progress_text.text("Stage 3: Running SBERT narrative semantic embedding...")
        bar.progress(55)
        t_sem_start = time.time()
        semantic_scores = compute_semantic_fit(filtered_candidates)
        t_sem = time.time() - t_sem_start
        
        # 4. Full Scoring
        progress_text.text("Stage 4: Extracting features, checking honeypots, and scoring...")
        bar.progress(75)
        t_score_start = time.time()
        scored_candidates = score_candidates(filtered_candidates, semantic_scores)
        t_score = time.time() - t_score_start
        
        # 5. MMR Reranking
        progress_text.text("Stage 5: Performing MMR diversity re-ranking...")
        bar.progress(90)
        t_mmr_start = time.time()
        top_scored = mmr_rerank(scored_candidates, top_n=100)
        top_scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        t_mmr = time.time() - t_mmr_start
        
        bar.progress(100)
        progress_text.text("Pipeline Execution Completed!")
        
        candidate_map = {c["candidate_id"]: c for c in filtered_candidates}
        
        # Build final display table
        display_rows = []
        for rank, scored in enumerate(top_scored, 1):
            cand = candidate_map[scored["candidate_id"]]
            fragility = compute_counterfactual_fragility(cand, scored)
            reason = generate_reasoning(cand, scored, rank, fragility)
            
            display_rows.append({
                "Rank": rank,
                "Candidate ID": scored["candidate_id"],
                "Score": round(scored["final_score"], 4),
                "Years Exp": cand["profile"]["years_of_experience"],
                "Location": cand["profile"]["location"],
                "Title": cand["profile"]["current_title"],
                "Company": next((h.get("company","Unknown") for h in cand.get("career_history",[]) if h.get("is_current")), cand.get("career_history",[{}])[0].get("company","Unknown")),
                "Reasoning": reason,
                "scored_data": scored,
                "raw_candidate": cand,
                "fragility_score": fragility
            })
            
        st.session_state.pipeline_results = {
            "display_df": pd.DataFrame(display_rows),
            "raw_results": display_rows,
            "metrics": {
                "total_candidates": len(all_candidates),
                "filtered_candidates": len(filtered_candidates),
                "time_load": t_load,
                "time_ret": t_ret,
                "time_sem": t_sem,
                "time_score": t_score,
                "time_mmr": t_mmr,
                "total_time": time.time() - t_start
            }
        }
        
    if st.session_state.pipeline_results is not None:
        res = st.session_state.pipeline_results
        df = res["display_df"]
        m = res["metrics"]
        
        st.success(f"Pipeline executed in **{m['total_time']:.2f}s** over {m['total_candidates']:,} candidates!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Input Candidates", f"{m['total_candidates']:,}")
        col2.metric("Post-BM25 Candidates", f"{m['filtered_candidates']:,}")
        col3.metric("Semantic Search Time", f"{m['time_sem']:.2f}s")
        col4.metric("Total Execution Time", f"{m['total_time']:.2f}s")
        
        st.subheader("Top 100 Ranked Candidates")
        st.dataframe(
            df[["Rank", "Candidate ID", "Score", "Years Exp", "Title", "Location", "Reasoning"]],
            use_container_width=True,
            hide_index=True
        )

# ── Tab 3: Analytics Dashboard ────────────────────────────────────────────────
with tab_analytics:
    if st.session_state.pipeline_results is None:
        st.info("Run the pipeline first to view analytics.")
    else:
        res = st.session_state.pipeline_results
        df = res["display_df"]
        
        st.subheader("Top 100 Candidate Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score Distribution
            fig_score = px.histogram(df, x="Score", nbins=15, 
                                     title="Distribution of Scores",
                                     color_discrete_sequence=["#FF4B4B"])
            fig_score.update_layout(bargap=0.1)
            st.plotly_chart(fig_score, use_container_width=True)
            
        with col2:
            # Years of Experience Distribution
            fig_yoe = px.histogram(df, x="Years Exp", nbins=10,
                                   title="Distribution of Years of Experience",
                                   color_discrete_sequence=["#FF8F8F"])
            fig_yoe.update_layout(bargap=0.1)
            st.plotly_chart(fig_yoe, use_container_width=True)
            
        col3, col4 = st.columns(2)
        
        with col3:
            # Title breakdown
            title_counts = df["Title"].value_counts().reset_index()
            title_counts.columns = ["Title", "Count"]
            fig_title = px.bar(title_counts.head(10), x="Count", y="Title", orientation="h",
                               title="Top 10 Titles in Ranked Pool",
                               color="Count", color_continuous_scale="Reds")
            st.plotly_chart(fig_title, use_container_width=True)
            
        with col4:
            # Location breakdown
            loc_counts = df["Location"].value_counts().reset_index()
            loc_counts.columns = ["Location", "Count"]
            fig_loc = px.pie(loc_counts.head(8), values="Count", names="Location",
                             title="Top Locations of Ranked Candidates",
                             color_discrete_sequence=px.colors.sequential.Reds)
            st.plotly_chart(fig_loc, use_container_width=True)

# ── Tab 4: Candidate Inspector ────────────────────────────────────────────────
with tab_inspector:
    if st.session_state.pipeline_results is None:
        st.info("Run the pipeline first to select and inspect candidates.")
    else:
        res = st.session_state.pipeline_results
        raw_list = res["raw_results"]
        
        # Candidate Selector
        candidate_options = {f"Rank {c['Rank']} — {c['Candidate ID']} ({c['Title']})": i for i, c in enumerate(raw_list)}
        selected_label = st.selectbox("Select Candidate to Inspect", options=list(candidate_options.keys()))
        selected_idx = candidate_options[selected_label]
        
        c = raw_list[selected_idx]
        cand = c["raw_candidate"]
        scored = c["scored_data"]
        
        st.markdown(f"### candidate: {c['Candidate ID']} (Rank {c['Rank']})")
        st.markdown(f"**Current Role**: {c['Title']} at **{c['Company']}** | **Location**: {c['Location']}")
        
        # Details layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Match Scoring Summary")
            
            # Big score display
            st.metric("Final Score", f"{scored['final_score']:.4f}", help="Base Score * Behavioral Multiplier")
            
            # Subscore metrics
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Semantic Fit (40%)", f"{scored['semantic_score']:.2f}")
            s_col2.metric("Structural Fit (30%)", f"{scored['structural_score']:.2f}")
            s_col3.metric("Skills Overlap (30%)", f"{scored['skills_score']:.2f}")
            
            st.metric("Behavioral Multiplier", f"{scored['behavioral_multiplier']:.2f}", help="Clamped between 0.3 and 1.0 based on activity signals")
            st.metric("Counterfactual Fragility", f"{c['fragility_score']:.2f}", help="Score drop when best skill is ablated. Higher = more fragile, lower = robust.")
            
        with col2:
            st.write("#### System Reasoning & Details")
            st.markdown(f'<div class="candidate-card"><strong>Explainer:</strong><br/>{c["Reasoning"]}</div>', unsafe_allow_html=True)
            
            # Notice period & Relocation
            sig = cand["redrob_signals"]
            st.write(f"- **Notice Period:** {sig.get('notice_period_days', 'Unknown')} days")
            st.write(f"- **Willing to Relocate:** {'Yes' if sig.get('willing_to_relocate', False) else 'No'}")
            st.write(f"- **GitHub Activity Score:** {sig.get('github_activity_score', 'N/A')}/100")
            st.write(f"- **Recruiter Response Rate:** {int(sig.get('recruiter_response_rate', 0.5)*100)}%")
            st.write(f"- **Profile Active Status:** Active {sig.get('last_active_date', 'N/A')}")
            
        # Breakdowns
        st.write("#### Score Breakdowns")
        b_col1, b_col2, b_col3 = st.columns(3)
        
        with b_col1:
            st.write("**Structural Breakdown**")
            st.json(scored["structural_breakdown"])
            
        with b_col2:
            st.write("**Skills Breakdown**")
            st.json(scored["skills_breakdown"])
            
        with b_col3:
            st.write("**Behavioral Breakdown**")
            st.json(scored["behavioral_breakdown"])
            
        # Career History
        st.write("#### Career History Narrative")
        for role in cand.get("career_history", []):
            st.markdown(f"**{role.get('title')}** at {role.get('company')} ({role.get('duration_months', 0)} months)")
            st.write(role.get("description"))
            st.write("---")
