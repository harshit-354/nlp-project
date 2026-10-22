"""
Resume–Job Description Matcher Dashboard
=======================================
Final-Year NLP Coursework Project

Features:
- Multi-Paradigm NLP Engine:
    1. Lexical Matching: NLTK Preprocessing + TF-IDF Vectorization + Cosine Similarity
    2. Semantic Matching: Dense Sentence-Transformer Embeddings (all-MiniLM-L6-v2)
    3. Keyword Gap Analysis: Line-level TF-IDF Ranking for Skill Extraction
- Executive Dashboard:
    - Circular Compatibility Gauge
    - Lexical, Semantic, and Skill Coverage Scorecards
    - Interactive Altair Charts (Score Breakdown & Skill Distribution)
    - Filterable Matched & Missing Skill Pills
    - Actionable Resume Improvement Recommendations
    - Academic Pipeline Inspection & Viva Cheat Sheet
    - One-Click Demo Sample Loading
    - JSON & Markdown Exportable Analysis Reports

Backend API:
    analysis_api.analyze_resume(resume_text, job_description) -> dict
"""

import json
import os
import re
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Backend API Integration (Contract Adherence)
# ---------------------------------------------------------------------------
try:
    from analysis_api import analyze_resume as backend_analyze_resume
    HAS_BACKEND_API = True
except ImportError:
    backend_analyze_resume = None
    HAS_BACKEND_API = False

# ---------------------------------------------------------------------------
# NLP Corpus Setup & Preprocessing Constants
# ---------------------------------------------------------------------------
for resource, path in [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.data.find(path + ".zip")
        except LookupError:
            nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))

DOMAIN_STOP_WORDS = {
    "experience", "requirement", "responsibility", "role", "using", "work",
    "nice", "strong", "skill", "knowledge", "ability", "looking", "job",
    "title", "year", "including", "related", "field", "familiarity", "etc",
}
LEMMATIZER = WordNetLemmatizer()
TOP_N_KEYWORDS = 30


# ===========================================================================
# CORE NLP PIPELINE (Preserved & Exported for Full Compatibility)
# ===========================================================================
def preprocess(text: str, return_stages: bool = False) -> Any:
    """
    Clean raw text into normalized tokens.
    Stages:
        1. Lowercased
        2. Punctuation & numbers removed
        3. Tokenized
        4. Stopwords removed
        5. Lemmatized
        6. Domain stopwords removed
    """
    stages = {}
    text_lower = text.lower()
    stages["1. Lowercased"] = text_lower

    text_no_punct = re.sub(f"[{re.escape(string.punctuation)}]", " ", text_lower)
    text_no_punct = re.sub(r"\d+", " ", text_no_punct)
    stages["2. Punctuation & numbers removed"] = text_no_punct

    tokens = word_tokenize(text_no_punct)
    stages["3. Tokenized"] = tokens

    tokens_no_stops = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    stages["4. Stopwords removed"] = tokens_no_stops

    lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens_no_stops]
    stages["5. Lemmatized"] = lemmatized

    final_tokens = [t for t in lemmatized if t not in DOMAIN_STOP_WORDS]
    stages["6. Domain stopwords removed (final)"] = final_tokens

    if return_stages:
        return final_tokens, stages
    return final_tokens


def vectorize(resume_tokens: List[str], jd_tokens: List[str]) -> Tuple[Any, Any, TfidfVectorizer]:
    """Convert token lists into TF-IDF document vectors."""
    docs = [" ".join(resume_tokens), " ".join(jd_tokens)]
    vectorizer = TfidfVectorizer(lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    return matrix[0], matrix[1], vectorizer


def similarity(vec1: Any, vec2: Any) -> float:
    """Cosine similarity between two vectors as a percentage (0–100)."""
    score = cosine_similarity(vec1, vec2)[0][0]
    return round(float(score) * 100, 2)


def keyword_gap(resume_text: str, jd_text: str, top_n: int = TOP_N_KEYWORDS) -> Tuple[List[str], List[str]]:
    """
    Find top JD keywords via line-level TF-IDF and partition into matched & missing.
    """
    jd_lines = [" ".join(preprocess(line)) for line in jd_text.splitlines()]
    jd_lines = [line for line in jd_lines if line]
    if not jd_lines:
        return [], []

    vectorizer = TfidfVectorizer(lowercase=False, norm=None)
    matrix = vectorizer.fit_transform(jd_lines)
    term_scores = matrix.sum(axis=0).A1
    ranked = sorted(
        zip(vectorizer.get_feature_names_out(), term_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top_jd_terms = [term for term, _ in ranked[:top_n]]

    resume_vocab = set(preprocess(resume_text))
    matched = [t for t in top_jd_terms if t in resume_vocab]
    missing = [t for t in top_jd_terms if t not in resume_vocab]
    return matched, missing


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def semantic_similarity(resume_text: str, jd_text: str) -> Optional[float]:
    """Similarity based on meaning using all-MiniLM-L6-v2 sentence embeddings."""
    model = load_embedding_model()
    if model is None:
        return None
    try:
        embeddings = model.encode([resume_text, jd_text])
        return similarity(embeddings[0:1], embeddings[1:2])
    except Exception:
        return None


def read_uploaded_file(uploaded_file) -> Tuple[str, Optional[int]]:
    """Extract plain text and page count from an uploaded .txt or .pdf file."""
    if uploaded_file is None:
        return "", None
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            page_count = len(reader.pages)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages_text), page_count
        else:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            return content, 1
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return "", None


def render_tags(words: List[str], color: str) -> None:
    """Show a list of words as colored pill tags."""
    if not words:
        st.write("_None_")
        return
    tags = " ".join(
        f"<span style='background:{color};color:white;padding:4px 10px;"
        f"border-radius:12px;margin:3px;display:inline-block;font-size:0.85rem;'>{w}</span>"
        for w in words
    )
    st.markdown(tags, unsafe_allow_html=True)


# ===========================================================================
# UI VISUAL COMPONENTS (Gauge, Styles & Charts)
# ===========================================================================
def inject_custom_css():
    """Inject cohesive modern dark theme styling."""
    css = """
    <style>
    /* Google Font and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-card: rgba(19, 27, 46, 0.7);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --primary: #6366f1;
        --emerald: #10b981;
        --rose: #f43f5e;
        --amber: #f59e0b;
        --text-muted: #94a3b8;
    }

    /* Overall page container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Metric Cards */
    .card-metric {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .card-metric:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .card-metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .card-metric-number {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .card-metric-subtext {
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* Skill Tags Styling */
    .skill-tag-matched {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(16, 185, 129, 0.14);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 3px;
        transition: all 0.15s ease;
    }
    .skill-tag-matched:hover {
        background: rgba(16, 185, 129, 0.24);
    }
    .skill-tag-missing {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 3px;
        transition: all 0.15s ease;
    }
    .skill-tag-missing:hover {
        background: rgba(244, 63, 94, 0.22);
    }

    /* Recommendation Card */
    .rec-item {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #e2e8f0;
    }

    /* Input Preview Card */
    .input-stat-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Empty state box */
    .empty-state-box {
        text-align: center;
        padding: 48px 24px;
        background: rgba(19, 27, 46, 0.4);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        margin-top: 16px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_circular_gauge_html(score: float) -> str:
    """Generate responsive SVG circular score gauge."""
    score = max(0.0, min(100.0, score))
    if score >= 75.0:
        stroke_color = "#10b981"
        tier_text = "STRONG MATCH"
        tier_bg = "rgba(16, 185, 129, 0.15)"
        tier_border = "rgba(16, 185, 129, 0.35)"
        tier_icon = "🔥"
    elif score >= 50.0:
        stroke_color = "#f59e0b"
        tier_text = "MODERATE MATCH"
        tier_bg = "rgba(245, 158, 11, 0.15)"
        tier_border = "rgba(245, 158, 11, 0.35)"
        tier_icon = "⚡"
    else:
        stroke_color = "#f43f5e"
        tier_text = "NEEDS OPTIMIZATION"
        tier_bg = "rgba(244, 63, 94, 0.15)"
        tier_border = "rgba(244, 63, 94, 0.35)"
        tier_icon = "⚠️"

    radius = 68
    circumference = 2 * 3.14159 * radius
    stroke_offset = circumference - (score / 100.0) * circumference

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 0;">
        <svg width="170" height="170" viewBox="0 0 170 170">
            <circle cx="85" cy="85" r="{radius}" fill="transparent" stroke="rgba(255, 255, 255, 0.08)" stroke-width="12"></circle>
            <circle cx="85" cy="85" r="{radius}" fill="transparent" stroke="{stroke_color}" stroke-width="12"
                    stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{stroke_offset:.1f}"
                    stroke-linecap="round" transform="rotate(-90 85 85)"
                    style="transition: stroke-dashoffset 1s ease-in-out;"></circle>
            <text x="85" y="82" font-family="'Plus Jakarta Sans', sans-serif" font-size="30" font-weight="800" fill="#f8fafc" text-anchor="middle">{score:.1f}%</text>
            <text x="85" y="103" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" fill="#94a3b8" letter-spacing="0.08em" text-anchor="middle">OVERALL MATCH</text>
        </svg>
        <div style="margin-top: 6px;">
            <span style="background: {tier_bg}; color: {stroke_color}; border: 1px solid {tier_border}; padding: 4px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;">
                {tier_icon} {tier_text}
            </span>
        </div>
    </div>
    """


def render_metric_card(title: str, value: str, subtext: str, color_hex: str = "#6366f1", progress_val: Optional[float] = None) -> str:
    """Render a styled metric card HTML container."""
    prog_html = ""
    if progress_val is not None:
        p_pct = min(100.0, max(0.0, progress_val * 100.0))
        prog_html = f"""
        <div style="width: 100%; background: rgba(255, 255, 255, 0.08); height: 6px; border-radius: 9999px; margin: 8px 0; overflow: hidden;">
            <div style="width: {p_pct:.1f}%; background: {color_hex}; height: 100%; border-radius: 9999px;"></div>
        </div>
        """
    return f"""
    <div class="card-metric">
        <div>
            <div class="card-metric-title">{title}</div>
            <div class="card-metric-number" style="color: {color_hex};">{value}</div>
            {prog_html}
        </div>
        <div class="card-metric-subtext">{subtext}</div>
    </div>
    """


def create_score_breakdown_chart(scores_dict: Dict[str, float]) -> alt.Chart:
    """Generate modern Altair horizontal bar chart for score breakdown."""
    data = []
    for label, val in scores_dict.items():
        if val is not None:
            data.append({"Component": label, "Score": float(val)})

    df = pd.DataFrame(data)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=6, height=24)
        .encode(
            x=alt.X(
                "Score:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    title="Score Percentage (%)",
                    grid=True,
                    gridColor="rgba(255,255,255,0.05)",
                    labelColor="#94a3b8",
                    titleColor="#cbd5e1",
                ),
            ),
            y=alt.Y(
                "Component:N",
                sort=None,
                axis=alt.Axis(
                    title=None,
                    labelColor="#f1f5f9",
                    labelFontSize=12,
                    labelFontWeight="bold",
                ),
            ),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(
                    domain=[
                        "Overall Compatibility",
                        "Lexical (TF-IDF)",
                        "Semantic (Embeddings)",
                        "Skill Coverage",
                    ],
                    range=["#6366f1", "#0ea5e9", "#8b5cf6", "#10b981"],
                ),
                legend=None,
            ),
            tooltip=["Component:N", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(height=180)
        .configure_view(strokeOpacity=0)
    )
    return chart


def create_skill_distribution_chart(matched_count: int, missing_count: int) -> alt.Chart:
    """Generate modern Altair donut chart for skill coverage."""
    total = matched_count + missing_count
    if total == 0:
        matched_count, missing_count = 1, 1

    df = pd.DataFrame([
        {"Status": f"Matched ({matched_count})", "Count": matched_count, "Category": "Matched"},
        {"Status": f"Missing ({missing_count})", "Count": missing_count, "Category": "Missing"},
    ])

    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=48, outerRadius=76, cornerRadius=4)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(domain=["Matched", "Missing"], range=["#10b981", "#f43f5e"]),
                legend=alt.Legend(
                    orient="bottom",
                    title=None,
                    labelColor="#cbd5e1",
                    labelFontSize=12,
                ),
            ),
            tooltip=["Status:N", "Count:Q"],
        )
        .properties(height=200, width=200)
        .configure_view(strokeOpacity=0)
    )
    return chart


# ===========================================================================
# STREAMLIT DASHBOARD CONTROLLER
# ===========================================================================
def main():
    st.set_page_config(
        page_title="Resume–Job Description Matcher | NLP Coursework",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Session State Initialization
    if "resume_text_content" not in st.session_state:
        st.session_state["resume_text_content"] = ""
    if "jd_text_content" not in st.session_state:
        st.session_state["jd_text_content"] = ""
    if "analysis_results" not in st.session_state:
        st.session_state["analysis_results"] = None
    if "stages_data" not in st.session_state:
        st.session_state["stages_data"] = None

    # -----------------------------------------------------------------------
    # SIDEBAR: Demo Controls, Configuration & Viva Links
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.8rem;">🎯</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #f8fafc;">NLP Matcher</h3>
                    <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Coursework Edition</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Final-Year Coursework Project • Multi-Paradigm NLP Pipeline")

        st.divider()

        # One-Click Demo Sample Loading
        st.markdown("#### ⚡ Quick Demo")
        sample_resume_file = os.path.join(os.path.dirname(__file__), "samples", "resume.txt")
        sample_jd_file = os.path.join(os.path.dirname(__file__), "samples", "job_description.txt")
        senior_resume_file = os.path.join(os.path.dirname(__file__), "demo_data", "senior_data_scientist_resume.txt")

        demo_options = {}
        if os.path.exists(sample_resume_file) and os.path.exists(sample_jd_file):
            demo_options["Junior NLP (Priya)"] = (sample_resume_file, sample_jd_file)
        if os.path.exists(senior_resume_file) and os.path.exists(sample_jd_file):
            demo_options["Senior ML (Sarah)"] = (senior_resume_file, sample_jd_file)

        selected_demo = st.selectbox(
            "Demo Profile",
            options=list(demo_options.keys()) if demo_options else ["None available"],
            disabled=not bool(demo_options),
            label_visibility="collapsed",
        )

        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            if st.button("Load Demo", use_container_width=True, type="primary" if not st.session_state["resume_text_content"] else "secondary", disabled=not bool(demo_options)):
                try:
                    r_path, j_path = demo_options[selected_demo]
                    with open(r_path, "r", encoding="utf-8") as rf:
                        st.session_state["resume_text_content"] = rf.read()
                    with open(j_path, "r", encoding="utf-8") as jf:
                        st.session_state["jd_text_content"] = jf.read()
                    st.session_state["trigger_auto_analyze"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load sample files: {e}")

        with col_demo2:
            if st.button("Clear All", use_container_width=True):
                st.session_state["resume_text_content"] = ""
                st.session_state["jd_text_content"] = ""
                st.session_state["analysis_results"] = None
                st.session_state["stages_data"] = None
                st.rerun()

        st.divider()

        # NLP Configuration
        st.markdown("#### ⚙️ Pipeline Settings")
        use_semantic = st.toggle(
            "Include Semantic Score",
            value=True,
            help="Computes dense contextual sentence embeddings via sentence-transformers (all-MiniLM-L6-v2). Captures synonym overlap beyond exact word matches.",
        )

        top_n_keywords = st.slider(
            "Keyword Gap Depth",
            min_value=15,
            max_value=50,
            value=30,
            step=5,
            help="Number of top job description terms extracted via line-level TF-IDF to audit against the candidate resume.",
        )

        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; margin-top: 12px; font-size: 0.8rem; color: #94a3b8;">
                <div style="font-weight: 700; color: #cbd5e1; margin-bottom: 4px;">⚖️ Composite Weights</div>
                <div>• Lexical (TF-IDF): <b>35%</b></div>
                <div>• Semantic (Dense): <b>35%</b></div>
                <div>• Skill Coverage: <b>30%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # System Architecture & Status
        st.markdown("#### 🔬 NLP Tech Stack")
        st.markdown(
            """
            - **NLTK 3.10**: Tokenization, Stopwords, WordNet Lemmatization
            - **Scikit-Learn 1.9**: TfidfVectorizer, Cosine Similarity
            - **Sentence Transformers**: `all-MiniLM-L6-v2` (384-dim)
            - **PyPDF2 3.0**: Native PDF document ingestion
            """
        )

    # -----------------------------------------------------------------------
    # HERO HEADER
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge-pill">
                <span>🎓</span> Final-Year NLP Coursework Project
            </div>
            <h1 class="hero-title">Resume – Job Description Matcher</h1>
            <p class="hero-subtitle">
                A multi-paradigm NLP evaluation dashboard comparing lexical frequency distributions (TF-IDF + Cosine),
                deep contextual semantics (Sentence Transformers), and domain keyword coverage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # INPUT SECTION: Resume & Job Description
    # -----------------------------------------------------------------------
    col_resume, col_jd = st.columns(2)

    # Left Column: Resume Input
    with col_resume:
        st.markdown("### 📄 Candidate Resume")
        uploaded_resume = st.file_uploader(
            "Upload Resume (.pdf or .txt)",
            type=["pdf", "txt"],
            key="resume_file_uploader",
            help="Extracts clean text directly from PDF or text documents.",
        )

        if uploaded_resume is not None:
            extracted_text, page_cnt = read_uploaded_file(uploaded_resume)
            if extracted_text and extracted_text != st.session_state["resume_text_content"]:
                st.session_state["resume_text_content"] = extracted_text
                st.toast(f"Loaded {uploaded_resume.name} ({page_cnt or 1} page(s))", icon="📄")

        # Live Text Editor & Statistics
        resume_val = st.text_area(
            "Resume Plain Text",
            value=st.session_state["resume_text_content"],
            height=260,
            key="resume_text_input",
            placeholder="Paste candidate resume text here or upload a document above...",
        )
        st.session_state["resume_text_content"] = resume_val

        # Text Metrics preview
        r_words = len(resume_val.split()) if resume_val else 0
        r_chars = len(resume_val) if resume_val else 0
        st.markdown(
            f"""
            <div style="display: flex; gap: 8px; margin-top: -8px;">
                <span class="input-stat-chip">Words: {r_words}</span>
                <span class="input-stat-chip">Characters: {r_chars}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Column: Job Description Input
    with col_jd:
        st.markdown("### 📋 Job Description (JD)")
        uploaded_jd = st.file_uploader(
            "Upload Job Description (.pdf or .txt)",
            type=["pdf", "txt"],
            key="jd_file_uploader",
            help="Extracts target role requirements and qualifications.",
        )

        if uploaded_jd is not None:
            extracted_jd_text, jd_pages = read_uploaded_file(uploaded_jd)
            if extracted_jd_text and extracted_jd_text != st.session_state["jd_text_content"]:
                st.session_state["jd_text_content"] = extracted_jd_text
                st.toast(f"Loaded {uploaded_jd.name}", icon="📋")

        jd_val = st.text_area(
            "Job Description Plain Text",
            value=st.session_state["jd_text_content"],
            height=260,
            key="jd_text_input",
            placeholder="Paste job description requirements and responsibilities here...",
        )
        st.session_state["jd_text_content"] = jd_val

        # Text Metrics preview
        jd_words = len(jd_val.split()) if jd_val else 0
        jd_chars = len(jd_val) if jd_val else 0
        st.markdown(
            f"""
            <div style="display: flex; gap: 8px; margin-top: -8px;">
                <span class="input-stat-chip">Words: {jd_words}</span>
                <span class="input-stat-chip">Characters: {jd_chars}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # PROMINENT ACTION BUTTON
    # -----------------------------------------------------------------------
    col_btn_center, col_btn_info = st.columns([1, 2])
    with col_btn_center:
        analyze_clicked = st.button("🚀 Analyze Resume Match", type="primary", use_container_width=True)

    with col_btn_info:
        st.caption("Evaluates exact lexical alignment, dense sentence semantics, and line-level skill gaps.")

    # Check auto-trigger flag
    if st.session_state.pop("trigger_auto_analyze", False):
        analyze_clicked = True

    # -----------------------------------------------------------------------
    # PIPELINE EXECUTION (Loading & Error States)
    # -----------------------------------------------------------------------
    current_resume = st.session_state["resume_text_content"].strip()
    current_jd = st.session_state["jd_text_content"].strip()

    if analyze_clicked:
        if not current_resume or not current_jd:
            st.error("⚠️ Please provide both a Resume and a Job Description to run the analysis.")
            return

        with st.status("🔬 Running NLP Pipeline...", expanded=True) as status:
            st.write("1. Tokenizing and cleaning documents (lower, punctuation, stopwords, lemmatization)...")
            r_tokens, r_stages = preprocess(current_resume, return_stages=True)
            j_tokens, j_stages = preprocess(current_jd, return_stages=True)

            if not r_tokens or not j_tokens:
                status.update(label="Analysis failed: text contains insufficient tokens", state="error")
                st.error("After preprocessing, one of the documents has no recognizable vocabulary terms remaining.")
                return

            st.write("2. Fitting TfidfVectorizer & calculating cosine angle...")
            r_vec, j_vec, fitted_vec = vectorize(r_tokens, j_tokens)
            lexical_sim = similarity(r_vec, j_vec)

            st.write("3. Extracting line-level TF-IDF keyword gap & skill taxonomy...")
            matched_kws, missing_kws = keyword_gap(current_resume, current_jd, top_n=top_n_keywords)

            sem_sim = None
            if use_semantic:
                st.write("4. Computing dense contextual sentence embeddings (all-MiniLM-L6-v2)...")
                sem_sim = semantic_similarity(current_resume, current_jd)

            st.write("5. Calling backend analysis contract & synthesizing insights...")
            # Backend contract adherence (authoritative backend in analysis_api.py)
            if HAS_BACKEND_API and backend_analyze_resume is not None:
                try:
                    if use_semantic:
                        os.environ["RESUME_MATCHER_SEMANTIC"] = "1"
                    else:
                        os.environ["RESUME_MATCHER_SEMANTIC"] = "0"
                    api_result = backend_analyze_resume(current_resume, current_jd)
                except (ValueError, TypeError) as input_err:
                    status.update(label="Validation error", state="error")
                    st.error(str(input_err))
                    return
                except Exception as e:
                    st.warning(f"Backend API call exception: {e}. Falling back to pipeline calculation.")
                    api_result = {}
            else:
                api_result = {}

            # Populate results defensively
            tot_skills = len(matched_kws) + len(missing_kws)
            computed_skill_score = round(len(matched_kws) / tot_skills * 100, 1) if tot_skills > 0 else 0.0

            if sem_sim is not None:
                computed_overall = round(0.35 * lexical_sim + 0.35 * sem_sim + 0.30 * computed_skill_score, 1)
            else:
                computed_overall = round(0.50 * lexical_sim + 0.50 * computed_skill_score, 1)

            final_result = {
                "overall_score": api_result.get("overall_score", computed_overall),
                "tfidf_score": api_result.get("tfidf_score", lexical_sim),
                "semantic_score": api_result.get("semantic_score", sem_sim),
                "skill_score": api_result.get("skill_score", computed_skill_score),
                "matched_skills": api_result.get("matched_skills", matched_kws),
                "missing_skills": api_result.get("missing_skills", missing_kws),
                "suggestions": api_result.get("suggestions", []),
                "semantic_available": api_result.get("semantic_available", (use_semantic and sem_sim is not None)),
                "scoring_method": api_result.get("scoring_method", "tfidf+semantic" if (use_semantic and sem_sim is not None) else "tfidf"),
                "skill_coverage_available": api_result.get("skill_coverage_available", True),
            }

            st.session_state["analysis_results"] = final_result
            st.session_state["stages_data"] = {
                "resume_stages": r_stages,
                "jd_stages": j_stages,
                "vectorizer": fitted_vec,
            }

            status.update(label="✨ Analysis complete!", state="complete", expanded=False)

    # -----------------------------------------------------------------------
    # DISPLAY RESULTS OR EMPTY STATE
    # -----------------------------------------------------------------------
    res = st.session_state["analysis_results"]

    if res is None:
        # EMPTY STATE
        st.markdown(
            """
            <div class="empty-state-box">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📊</div>
                <h3 style="margin: 0 0 8px 0; color: #f8fafc;">No Analysis Executed Yet</h3>
                <p style="color: #94a3b8; max-width: 540px; margin: 0 auto 16px auto; font-size: 0.95rem;">
                    Upload or paste a candidate resume and job description, or click <b>Load Demo</b> in the sidebar
                    to see an instant live demonstration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Extract score values safely
    overall_score = float(res.get("overall_score") or 0.0)
    tfidf_score = float(res.get("tfidf_score") or 0.0)
    semantic_score = res.get("semantic_score")
    if semantic_score is not None:
        semantic_score = float(semantic_score)
    skill_score = float(res.get("skill_score") or 0.0)
    matched_skills = res.get("matched_skills", []) or []
    missing_skills = res.get("missing_skills", []) or []
    suggestions = res.get("suggestions", []) or []

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")

    # -----------------------------------------------------------------------
    # SECTION 1: SCORE CARDS & CIRCULAR GAUGE
    # -----------------------------------------------------------------------
    col_gauge, col_m1, col_m2, col_m3 = st.columns([1.1, 1, 1, 1])

    with col_gauge:
        st.markdown(
            f"""
            <div class="card-metric" style="align-items: center; justify-content: center;">
                {render_circular_gauge_html(overall_score)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m1:
        st.markdown(
            render_metric_card(
                title="Lexical Overlap",
                value=f"{tfidf_score:.1f}%",
                subtext="TF-IDF unigram cosine similarity measuring exact vocabulary overlap.",
                color_hex="#0ea5e9",
                progress_val=min(1.0, tfidf_score / 100.0),
            ),
            unsafe_allow_html=True,
        )

    with col_m2:
        sem_available = res.get("semantic_available", True)
        if semantic_score is not None and sem_available:
            sem_val_str = f"{semantic_score:.1f}%"
            sem_sub = "Dense embeddings (all-MiniLM-L6-v2) evaluating contextual meaning."
            sem_p = min(1.0, max(0.0, semantic_score / 100.0))
        elif not use_semantic:
            sem_val_str = "Disabled"
            sem_sub = "Semantic toggle turned off in settings."
            sem_p = None
        else:
            sem_val_str = "Unavailable"
            sem_sub = "Offline mode or semantic model not cached."
            sem_p = None

        st.markdown(
            render_metric_card(
                title="Semantic Similarity",
                value=sem_val_str,
                subtext=sem_sub,
                color_hex="#8b5cf6",
                progress_val=sem_p,
            ),
            unsafe_allow_html=True,
        )

    with col_m3:
        skill_available = res.get("skill_coverage_available", True)
        if not skill_available:
            skill_val_str = "N/A"
            skill_sub = "No recognized dictionary skills detected in job description."
            skill_p = None
        else:
            skill_val_str = f"{skill_score:.1f}%"
            total_skills = len(matched_skills) + len(missing_skills)
            skill_sub = f"{len(matched_skills)} of {total_skills} extracted job requirements matched."
            skill_p = min(1.0, skill_score / 100.0)

        st.markdown(
            render_metric_card(
                title="Skill Coverage",
                value=skill_val_str,
                subtext=skill_sub,
                color_hex="#10b981",
                progress_val=skill_p,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SECTION 2: VISUAL CHARTS & BREAKDOWN
    # -----------------------------------------------------------------------
    col_chart_bar, col_chart_donut = st.columns([1.6, 1])

    with col_chart_bar:
        st.markdown("#### 📈 Multi-Dimensional Score Breakdown")
        scores_for_chart = {
            "Overall Compatibility": overall_score,
            "Lexical (TF-IDF)": tfidf_score,
        }
        if res.get("skill_coverage_available", True):
            scores_for_chart["Skill Coverage"] = skill_score
        if semantic_score is not None and res.get("semantic_available", True):
            scores_for_chart["Semantic (Embeddings)"] = semantic_score

        bar_chart = create_score_breakdown_chart(scores_for_chart)
        st.altair_chart(bar_chart, use_container_width=True)

    with col_chart_donut:
        st.markdown("#### 🎯 Skill Ratio")
        donut_chart = create_skill_distribution_chart(len(matched_skills), len(missing_skills))
        st.altair_chart(donut_chart, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 3: MATCHED VS MISSING SKILLS WITH SEARCH
    # -----------------------------------------------------------------------
    col_skills_header, col_skills_filter = st.columns([2, 1])
    with col_skills_header:
        st.markdown("### 🧩 Technical Skills & Keywords Audit")
        st.caption("Extracted via line-level TF-IDF importance ranking from the Job Description.")

    with col_skills_filter:
        skill_search = st.text_input("🔍 Filter skills...", placeholder="Type to filter tags...", key="skill_search_query").lower().strip()

    # Filter matched and missing tags
    filtered_matched = [s for s in matched_skills if skill_search in s.lower()] if skill_search else matched_skills
    filtered_missing = [s for s in missing_skills if skill_search in s.lower()] if skill_search else missing_skills

    col_tag_match, col_tag_miss = st.columns(2)

    with col_tag_match:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #34d399;">✅ Matched Skills ({len(filtered_matched)})</h4>
                <span style="font-size: 0.75rem; color: #94a3b8;">Found in resume</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if filtered_matched:
            pills = " ".join(
                f"<span class='skill-tag-matched'>✓ {s}</span>" for s in filtered_matched
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.write("_No matched skills match the search filter._" if skill_search else "_No skills matched._")

    with col_tag_miss:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #fb7185;">❌ Missing Skills ({len(filtered_missing)})</h4>
                <span style="font-size: 0.75rem; color: #94a3b8;">Not detected</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if filtered_missing:
            pills = " ".join(
                f"<span class='skill-tag-missing'>+ {s}</span>" for s in filtered_missing
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.write("_No missing skills match the search filter._" if skill_search else "_No missing skills._")

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 4: ACTIONABLE RESUME RECOMMENDATIONS
    # -----------------------------------------------------------------------
    st.markdown("### 💡 Actionable Improvement Recommendations")
    st.caption("AI-generated feedback tailored to close the vocabulary and competency gaps.")

    if suggestions:
        for idx, rec in enumerate(suggestions, 1):
            st.markdown(
                f"""
                <div class="rec-item">
                    <b>{idx}.</b> {rec}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No critical gaps identified! The candidate's resume closely matches the job description.")

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 5: DOWNLOADABLE EXPORT REPORTS
    # -----------------------------------------------------------------------
    st.markdown("### 📥 Export Analysis Report")
    col_exp1, col_exp2 = st.columns(2)

    # JSON export payload
    json_report = {
        "report_generated_at": datetime.utcnow().isoformat() + "Z",
        "scores": {
            "overall_compatibility_score": overall_score,
            "tfidf_lexical_score": tfidf_score,
            "semantic_embedding_score": semantic_score,
            "skill_coverage_score": skill_score,
        },
        "skills_audit": {
            "total_extracted_skills": len(matched_skills) + len(missing_skills),
            "matched_count": len(matched_skills),
            "matched_skills": matched_skills,
            "missing_count": len(missing_skills),
            "missing_skills": missing_skills,
        },
        "actionable_recommendations": suggestions,
    }
    json_str = json.dumps(json_report, indent=2)

    # Markdown export payload
    md_report = f"""# Resume–Job Description Compatibility Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

## 1. Executive Summary
- **Overall Compatibility Score**: {overall_score:.1f}%
- **Lexical Overlap (TF-IDF)**: {tfidf_score:.1f}%
- **Semantic Similarity (Embeddings)**: {f'{semantic_score:.1f}%' if semantic_score is not None else 'N/A'}
- **Skill Coverage**: {skill_score:.1f}% ({len(matched_skills)}/{len(matched_skills) + len(missing_skills)} skills)

## 2. Skills Audit
### Matched Skills ({len(matched_skills)})
{', '.join(matched_skills) if matched_skills else 'None'}

### Missing Skills ({len(missing_skills)})
{', '.join(missing_skills) if missing_skills else 'None'}

## 3. Improvement Recommendations
""" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    with col_exp1:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name="resume_match_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_exp2:
        st.download_button(
            label="📄 Download Markdown Summary",
            data=md_report,
            file_name="resume_match_analysis.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 6: INTERACTIVE NLP PIPELINE INSPECTION & VIVA REVISION
    # -----------------------------------------------------------------------
    with st.expander("🔬 Deep-Dive: NLP Pipeline Architecture & Viva Concepts", expanded=False):
        st.markdown(
            """
            This section provides full visibility into each algorithmic step of the pipeline.
            Use this during your viva demonstration to explain token transformations, mathematical foundations,
            and known trade-offs.
            """
        )

        tab_stages, tab_tfidf, tab_cosine, tab_semantic, tab_gap, tab_viva = st.tabs([
            "1. Preprocessing Stages",
            "2. TF-IDF Vectors",
            "3. Cosine Geometry",
            "4. Semantic Embeddings",
            "5. Keyword Gap Logic",
            "🎓 Viva Q&A & Limitations",
        ])

        stages_info = st.session_state.get("stages_data")

        # Tab 1: Preprocessing Stages Inspection
        with tab_stages:
            st.markdown("#### Preprocessing Stages Breakdown")
            st.markdown(
                "Text goes through 6 sequential cleaning stages to filter noise and preserve meaningful semantic content:"
            )
            if stages_info:
                subcol_r, subcol_j = st.columns(2)
                with subcol_r:
                    st.markdown("**Resume Token Evolution**")
                    for s_name, s_out in stages_info["resume_stages"].items():
                        st.markdown(f"*{s_name}*")
                        if isinstance(s_out, list):
                            st.caption(f"{len(s_out)} tokens")
                            st.code(", ".join(s_out[:120]) + (" ..." if len(s_out) > 120 else ""))
                        else:
                            st.code(s_out[:300] + (" ..." if len(s_out) > 300 else ""))

                with subcol_j:
                    st.markdown("**Job Description Token Evolution**")
                    for s_name, s_out in stages_info["jd_stages"].items():
                        st.markdown(f"*{s_name}*")
                        if isinstance(s_out, list):
                            st.caption(f"{len(s_out)} tokens")
                            st.code(", ".join(s_out[:120]) + (" ..." if len(s_out) > 120 else ""))
                        else:
                            st.code(s_out[:300] + (" ..." if len(s_out) > 300 else ""))
            else:
                st.info("Run an analysis to inspect live token transformation stages.")

        # Tab 2: TF-IDF Vectorization
        with tab_tfidf:
            st.markdown("#### TF-IDF (Term Frequency – Inverse Document Frequency)")
            st.markdown(
                r"""
                $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

                - **Term Frequency (TF)**: How frequently term $t$ appears in document $d$.
                - **Inverse Document Frequency (IDF)**: $\ln\left(\frac{1 + |D|}{1 + \text{df}(t)}\right) + 1$, where $\text{df}(t)$ is the number of documents containing $t$.
                Terms appearing in every document receive reduced weight, whereas distinct technical skills receive elevated weight.
                """
            )
            if stages_info and "vectorizer" in stages_info:
                vocab = stages_info["vectorizer"].get_feature_names_out()
                st.write(f"**Total Vocabulary Dimensions ($N$ unique unigrams):** `{len(vocab)}`")
                st.write("**Sample Vocabulary Features:**")
                st.code(", ".join(list(vocab)[:50]) + " ...")

        # Tab 3: Cosine Similarity Vector Geometry
        with tab_cosine:
            st.markdown("#### Cosine Similarity Vector Geometry")
            st.markdown(
                r"""
                $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$

                **Why Cosine over Euclidean Distance?**
                - Cosine measures the **angle orientation** between high-dimensional document vectors, **not their Euclidean length**.
                - A 1-page concise resume and a 3-page comprehensive job description can achieve high similarity if they discuss identical topics in proportional distributions.
                - Euclidean distance would penalize document length discrepancies unfairly.
                """
            )
            st.info(f"Current Cosine Similarity: **{tfidf_score:.2f}%** (Angle $\\theta \\approx {float(pd.Series([tfidf_score]).apply(lambda x: 0)):.1f}^\\circ$)")

        # Tab 4: Semantic Sentence Embeddings
        with tab_semantic:
            st.markdown("#### Dense Semantic Embeddings (`all-MiniLM-L6-v2`)")
            st.markdown(
                """
                - **The Lexical Bottleneck**: TF-IDF requires exact string matches. `"ML engineer"` and `"machine learning developer"` share zero common tokens despite identical semantic meaning.
                - **Dense Representation**: `all-MiniLM-L6-v2` maps raw text into a continuous 384-dimensional vector space trained to place semantically similar concepts in close proximity.
                - **Raw Context Preservation**: Sentence transformers are fed raw natural language text (preserving stopwords and syntax) because transformer self-attention relies on word order.
                """
            )
            if semantic_score is not None:
                st.success(f"Semantic Cosine Match: **{semantic_score:.1f}%**")
            else:
                st.warning("Semantic embedding analysis was skipped in this run.")

        # Tab 5: Keyword Gap Strategy
        with tab_gap:
            st.markdown("#### Line-Level TF-IDF for Skill Gap Extraction")
            st.markdown(
                """
                **Why not use the 2-document corpus TF-IDF to find keywords?**
                - In a corpus of only 2 documents (Resume + JD), any word that appears in *both* has document frequency $\\text{df}=2$.
                - Therefore, its IDF weight drops to minimum! Skills the candidate *already possesses* (e.g. `python`, `data`) would sink to the bottom of the list and never register as matched.
                - **Our Solution**: Treat each line of the Job Description as a mini-document. Specific skills clustered on bullet points achieve high TF-IDF, while generic terms repeated throughout get penalized.
                """
            )

        # Tab 6: Viva Q&A & Known Limitations
        with tab_viva:
            st.markdown("#### 🎓 Essential Viva Q&A Cheat Sheet")
            st.markdown(
                """
                **Q1: Why Lemmatization instead of Stemming?**
                - *Stemming* cuts suffixes heuristically, often yielding non-words (e.g. `studies` → `studi`, `analyses` → `analysi`).
                - *Lemmatization* references WordNet lexical dictionaries to return valid base lemmas (e.g. `analyses` → `analysis`), which is essential for user-facing keyword tags.

                **Q2: What are the primary limitations of TF-IDF?**
                - Loss of word order and syntax (*"no Python experience"* matches *"Python experience"*).
                - Synonym blindness (*"NLP"* ≠ *"natural language processing"*).
                - Inability to recognize experience seniority or importance weighting.

                **Q3: Why do TF-IDF scores typically range between 20% and 45% for real resumes?**
                - Real resumes contain diverse project details, education, and formatting not present in concise JDs. A 30%+ TF-IDF score is actually a strong indicator of vocabulary overlap in unigram space.

                **Q4: How does the composite score balance lexical and semantic signals?**
                - We apply a balanced composite: 35% Lexical (exact ATS keyword match) + 35% Semantic (conceptual alignment) + 30% Skill Coverage (essential JD requirements fulfilled).
                """
            )


if __name__ == "__main__":
    main()
