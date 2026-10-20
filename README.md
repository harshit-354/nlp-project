# Resume–Job Description Matcher

**Live demo:** [Open the Streamlit app](https://nlp-project-gjwzmcdivpkazqvsxeathz.streamlit.app/)

A locally runnable NLP coursework dashboard for comparing a resume with a job
description. The dark Streamlit interface includes a circular compatibility
gauge, scorecards, Altair charts, skill filters, sample loader, JSON/Markdown
exports and an explorable NLP pipeline. Every result is computed from the input.
No API keys, paid services, corpus downloads or model downloads are required.

## Install and launch

Use Python 3.10+ (verified with Python 3.12). From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate`. Open http://localhost:8501.
Dependency installation requires internet or a prepared local package cache;
normal analysis runs offline after installation.

Select a profile in **Quick Demo** and click **Load Demo** for immediate analysis,
or upload a text-based PDF / UTF-8 TXT file or paste both documents, then click
**Analyze Resume Match**. All five demo choices use bundled illustrative sample
profiles and JDs; their analysis scores are real calculations. Changing input
invalidates the previous result. **Clear All** clears inputs and results.

## NLP and scoring

`analysis_api.py` is the authoritative engine. The dashboard uses its real
preprocessing stages and results, with no substitute scoring implementation.

1. Normalize Unicode and case, canonicalize dictionary skill aliases, tokenize
   and remove bundled English/domain stopwords. No stemming or lemmatization.
2. Fit TF-IDF jointly on the two documents and compute cosine similarity × 100.
3. Extract complete skill names and multi-word aliases from `skills.json`.
   Matched skills occur in both texts; missing skills occur only in the JD.
4. Compute skill coverage = matched JD skills / detected JD skills × 100.
5. Optionally compute local sentence embeddings, or fall back automatically.
6. Return weighted compatibility and actionable, conditional recommendations.

| Mode | Overall score |
|---|---|
| Offline / semantic unavailable | 60% TF-IDF + 40% skill coverage |
| Semantic available | 35% TF-IDF + 25% semantic + 40% skill coverage |
| No dictionary skills in JD | Remaining text weights renormalized |

The API returns floats from 0–100 for `overall_score`, `tfidf_score`,
`semantic_score`, `skill_score`, plus string lists `matched_skills`,
`missing_skills`, `suggestions`. `semantic_available=False` marks the numeric
semantic zero as unavailable; the UI and exports label this honestly. Likewise,
`skill_coverage_available=False` means no dictionary skills, not measured zero
coverage. JSON exports use null for unavailable component scores.

```python
from analysis_api import analyze_resume
result = analyze_resume("Python SQL projects", "Python SQL Docker")
```

See [BACKEND_API.md](BACKEND_API.md) for the full contract. The optional keyword
`use_semantic=True/False` controls an individual request without changing global
environment settings; the original two-argument API remains supported.

## Optional semantic model

The default demo runs without the heavy semantic dependencies. To prepare
semantic inference explicitly before a demonstration:

```bash
python -m pip install -r requirements-semantic.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

The second command downloads a pretrained model and needs internet. It is never
run automatically by the app. Enable **Include Semantic Score** when the model
is cached. `RESUME_MATCHER_MODEL` can point to a local model directory. Direct
API calls can also opt in with `RESUME_MATCHER_SEMANTIC=1`. Load failures are
cached; restart after installing a missing model. Inference uses CPU and averages
180-word chunk embeddings. Any model/import/inference failure falls back to
TF-IDF without breaking the application.

## Files and verification

- `app.py`: premium Streamlit dashboard.
- `analysis_api.py`, `skills.json`: NLP engine and editable skill aliases.
- `samples/`, `demo_data/`: bundled sample inputs, all wired into the demo loader.
- `docs/DEMO_GUIDE.md`: presentation walkthrough.
- `docs/VIVA_NOTES.md`: 24 technical questions with answers.
- `tests/test_analysis_api.py`: offline unit and PDF-to-analysis tests.
- `smoke_check.py`: API contract and headless Streamlit interaction checks.
- `demo_data/scripts/smoke_test.sh`: dependency, sample and documentation checks.

```bash
python -m unittest discover -s tests -v
python smoke_check.py
bash demo_data/scripts/smoke_test.sh
```

## Limitations

Scores are heuristic text comparisons, not probabilities or hiring decisions.
The English-oriented skill dictionary is finite; aliases can be ambiguous, and
skill mentions do not prove proficiency or account for negation. Unknown skills
can still contribute to TF-IDF. Scanned PDFs require external OCR. Encrypted
PDFs, PDFs over 100 pages, and texts over 100,000 characters are rejected with
an actionable error. Empty/stopword-only API inputs return zero scores; the UI
requests meaningful text. Files are processed locally and are not written to
application storage. No training from scratch is performed.
