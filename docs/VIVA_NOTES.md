# Viva Notes: 24 Technical Questions and Answers

### 1. What problem does the application solve?
It compares supplied resume and JD text, estimates lexical/skill compatibility,
and identifies mentioned and missing dictionary skills for manual review.

### 2. What is the authoritative entry point?
`analysis_api.analyze_resume(resume_text, job_description)` returns scores,
skill lists, recommendations and availability metadata. Streamlit renders it.

### 3. How is PDF text extracted?
`pypdf.PdfReader` extracts text page by page. Invalid, password-protected,
image-only and over-100-page documents raise actionable errors. There is no OCR.

### 4. What normalization is used?
Unicode NFKC, case folding, dash normalization and whitespace normalization.
This makes full-width Python and ordinary Python comparable.

### 5. How is tokenization performed?
A Unicode-aware regular expression extracts word tokens after skill aliases
are replaced with canonical placeholders. Numbers alone are not tokens.

### 6. Which stopwords are removed?
Scikit-learn's bundled English stopwords and generic resume/JD words such as
experience and requirements. No network corpus is needed.

### 7. Does the final application lemmatize or stem?
No. It uses explicit skill aliases and stopword filtering. Avoiding a lexical
corpus improves offline reliability, but generic inflections may remain distinct.

### 8. What is TF-IDF?
Term frequency multiplied by inverse document frequency. The vectorizer uses
smoothed IDF: log((1 + document count)/(1 + document frequency)) + 1, then L2
normalizes vectors. Terms common to both documents receive lower IDF.

### 9. On which corpus is TF-IDF fitted?
The current resume and JD together, as two documents. It is not fitted on JD
lines. This is a small comparison corpus, not a trained hiring model.

### 10. What is cosine similarity?
The dot product divided by the product of vector norms. It compares direction.
Nonnegative TF-IDF vectors give cosine in 0–1, multiplied by 100 for display.

### 11. What does a score of 70 mean?
It is a heuristic comparison score. It does not mean a 70% chance of being hired,
70% accuracy, or a calibrated probability of candidate suitability.

### 12. How are skills extracted?
Boundary-aware regular expressions match canonical names and aliases in
`skills.json`, including multi-word phrases. Results are sorted and deduplicated.

### 13. How do aliases improve matching?
For example, NLP and natural language processing map to one canonical skill.
Placeholders preserve punctuation-heavy skills such as C++ and C# in TF-IDF.

### 14. How are matched and missing skills determined?
Matched = JD skill set intersect resume skill set. Missing = JD skill set minus
resume skill set. Skill coverage = matched count / JD skill count × 100.

### 15. How is the overall score weighted?
Offline: 60% TF-IDF + 40% skills. Semantic available: 35% TF-IDF + 25% semantic
+ 40% skills. These explicit heuristic weights have not been statistically tuned.

### 16. What if the JD contains no dictionary skills?
Skill coverage is unavailable and the remaining text weights are renormalized.
The UI shows N/A instead of suggesting that zero skill coverage was measured.

### 17. Which semantic model is supported?
The default is pretrained sentence-transformers/all-MiniLM-L6-v2. It produces
384-dimensional embeddings. A local compatible model path can be configured.

### 18. What happens when the semantic model is missing?
Import, loading or inference failures return unavailable metadata and trigger
TF-IDF fallback. Runtime loading is local-only, so no model download is attempted.

### 19. What is cached?
Compiled dictionary patterns, loaded semantic models and model-load failures.
Resume text and results are kept in the current Streamlit session, not a shared
analysis cache. Restart after preparing a previously missing model.

### 20. How are long documents handled?
Texts are limited to 100,000 characters. Semantic processing uses 180-word chunks
and averages embeddings; the model may still truncate token-heavy chunks.
TF-IDF uses the full accepted preprocessed document.

### 21. What happens with empty or malformed input?
Empty or stopword-only API inputs return zero scores and a useful suggestion.
Non-string inputs raise TypeError, oversized inputs raise ValueError, and invalid
PDFs raise actionable ValueError messages. The UI catches validation errors.

### 22. How are recommendations generated?
Deterministic rules reference detected missing skills and low lexical overlap.
They advise adding truthful project evidence or learning the skill, not inventing
experience. No generative API creates the recommendations.

### 23. How was the application tested?
Backend unit tests cover aliases, score bounds, Unicode, empty input, PDF
extraction and semantic fallback. Streamlit AppTest exercises demo loading,
analysis, controls and exports. A live headless server health check verifies startup.

### 24. What are the main limitations and possible improvements?
The dictionary is English-oriented and finite; ambiguous terms and negation can
cause errors. Text mentions do not measure proficiency or seniority. OCR,
multilingual processing and evaluated scoring calibration are future work,
not capabilities claimed by this coursework demonstration.
