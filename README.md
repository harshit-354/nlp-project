# Resume–Job Description Matcher (NLP Coursework Project)

**Live demo:** [Open the Streamlit app](https://nlp-project-gjwzmcdivpkazqvsxeathz.streamlit.app/)

A production-grade NLP matching dashboard comparing candidate resumes with job descriptions using a multi-paradigm approach:

1. **Overall Compatibility Score**: Composite metric balancing lexical, semantic, and skill coverage metrics.
2. **Lexical Match (TF-IDF + Cosine)**: Exact unigram vocabulary overlap.
3. **Semantic Match (Sentence Transformers)**: Dense 384-dimensional embeddings (`all-MiniLM-L6-v2`) capturing conceptual relevance and synonyms.
4. **Skill Coverage Audit**: Line-level TF-IDF keyword extraction categorizing skills into Matched and Missing colored tags.
5. **Actionable Improvement Recommendations**: Contextual resume enhancement advice.
6. **Viva Demonstration Deep-Dive**: Live inspection of text across all 6 preprocessing stages, TF-IDF matrix geometry, and mathematical derivations.

## Project Structure

```
├── .streamlit/
│   └── config.toml             # Custom modern dark theme configuration
├── analysis_api.py             # Root NLP backend API (contract: analyze_resume)
├── app.py                      # Streamlit frontend dashboard & pipeline controller
├── requirements.txt            # Project dependencies
├── smoke_check.py              # Automated verification & contract smoke check
├── samples/
│   ├── resume.txt              # Sample junior NLP engineer resume
│   └── job_description.txt    # Target junior NLP engineer job description
├── demo_data/
│   └── senior_data_scientist_resume.txt  # Comparison senior resume
└── VIVA_CHEAT_SHEET.md         # Key concepts, viva questions & limitations
```

## Setup & Installation

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

On first launch, required NLTK corpora (`punkt`, `stopwords`, `wordnet`) are downloaded automatically.

## Running the Dashboard

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501`:
1. Use the **⚡ Quick Demo** selector in the sidebar to load candidate samples in one click, or upload custom `.pdf` / `.txt` files.
2. Click **🚀 Analyze Resume Match**.
3. Inspect the circular compatibility gauge, metric scorecards, Altair visual analytics charts, and filterable skill chips.
4. Download structured reports in **JSON** or **Markdown** formats.
5. Open the **🔬 Deep-Dive: NLP Pipeline Architecture & Viva Concepts** expander for demonstration walkthroughs.

## Backend Contract (`analysis_api.py`)

The root-level `analysis_api.py` exposes:

```python
analyze_resume(resume_text: str, job_description: str) -> dict
```

### Response Schema:
| Key | Type | Description |
|---|---|---|
| `overall_score` | `float` (0–100) | Weighted composite compatibility score |
| `tfidf_score` | `float` (0–100) | Lexical cosine similarity percentage |
| `semantic_score` | `float` (0–100) / `None` | Dense sentence embedding similarity |
| `skill_score` | `float` (0–100) | Ratio of matched top job keywords |
| `matched_skills` | `list[str]` | Keywords present in both resume and JD |
| `missing_skills` | `list[str]` | Top JD keywords missing from the resume |
| `suggestions` | `list[str]` | Actionable improvement recommendations |

## Automated Smoke Check

Verify system integrity and backend contract compliance with:

```bash
python smoke_check.py
```
