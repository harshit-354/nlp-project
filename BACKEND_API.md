# Backend integration contract

```python
from analysis_api import analyze_resume, read_uploaded_file
result = analyze_resume(resume_text, job_description)
```

Required fields: `overall_score`, `tfidf_score`, `semantic_score`, `skill_score`
are floats in 0–100; `matched_skills`, `missing_skills`, `suggestions` are lists
of strings. Results are calculated from the supplied documents, never mocked.

Additional fields:
- `semantic_available`: display semantic score as **Unavailable** when false.
  The numeric zero is a compatibility sentinel, not measured similarity.
- `scoring_method`: `tfidf` or `tfidf+semantic`.
- `skill_coverage_available`: false when no dictionary skills occur in the JD;
  display skill coverage as **N/A** in this case.

Overall weighting is 60% TF-IDF + 40% skill coverage in offline mode.
When semantic inference succeeds, use 35% TF-IDF + 25% semantic + 40% skills.
With no JD skills, text weights are renormalized. Skill coverage is matched
JD dictionary skills divided by all detected JD dictionary skills. Empty or
stopword-only documents return zero scores and an actionable suggestion.
Non-string inputs raise TypeError; documents over 100,000 characters raise
ValueError. The UI should display these errors without a traceback.

`extract_pdf_text(bytes_or_binary_file_or_path)` extracts text from PDFs.
`read_uploaded_file(uploaded_file)` handles Streamlit PDF/TXT uploads. Both
raise actionable ValueError messages for invalid files. Password-protected,
image-only and over-100-page PDFs are rejected. OCR is outside scope.

Preprocessing uses Unicode normalization, case folding, canonical skill
aliases, Unicode tokenization and scikit-learn's bundled English stopwords.
No NLTK downloads or API keys are needed. `preprocess(text, return_stages=True)`
returns `(tokens, stages)` for a pipeline explainer. Edit `skills.json` to add
canonical names and aliases. Matches use token boundaries; mentions do not
prove proficiency or infer negation. Short ambiguous names can be false positives.

Install core dependencies with `python -m pip install -r requirements.txt`.
Optional semantic setup: install `requirements-semantic.txt`, explicitly cache
`sentence-transformers/all-MiniLM-L6-v2` before the demo, and set
`RESUME_MATCHER_SEMANTIC=1`. `RESUME_MATCHER_MODEL` can point to a local model.
The app never downloads a model automatically. Model loading and failures are
cached; restart after installing a previously missing model. Missing packages,
uncached models and inference failures fall back automatically. Mean-pooled
180-word chunks cover long documents, although the model can further truncate
token-heavy chunks. The scores are heuristics, not hiring probabilities.

Run the automated offline check:

```bash
python -m unittest discover -s tests -v
```

The optional keyword-only `use_semantic=True/False` overrides the environment
for one analysis request. The dashboard uses it to keep each session's toggle
independent, without modifying process-wide environment variables.
