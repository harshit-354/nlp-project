"""Offline-first resume matching API; no UI imports or runtime downloads.

Scores are heuristic text comparisons, not hiring predictions. Skills are
mentions, not verified proficiency. Semantic inference is opt-in via
RESUME_MATCHER_SEMANTIC=1 and only loads an already cached/local model.
"""
from functools import lru_cache
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_TEXT_LENGTH = 100_000
DOMAIN_STOP_WORDS = frozenset('experience experiences requirement requirements responsibility responsibilities role using work strong skill skills knowledge ability looking job title years year including related field familiarity'.split())


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError('Resume and job description must be strings.')
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f'Each document must contain at most {MAX_TEXT_LENGTH:,} characters.')
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    return re.sub(r'\s+', ' ', text).strip()


@lru_cache(maxsize=1)
def _skill_patterns():
    dictionary = json.loads(Path(__file__).with_name('skills.json').read_text(encoding='utf-8'))
    patterns = []
    for canonical, aliases in dictionary.items():
        alternatives = []
        for alias in [canonical, *aliases]:
            escaped = re.escape(normalize_text(alias)).replace(r'\ ', r'[\s\-]+')
            alternatives.append(escaped)
        patterns.append((canonical, re.compile(r'(?<![\w+#])(?:' + '|'.join(sorted(alternatives, key=len, reverse=True)) + r')(?![\w+#])')))
    return tuple(patterns)


def extract_skills(text: str) -> list[str]:
    """Return sorted canonical skill mentions using boundary-aware aliases."""
    normalized = normalize_text(text)
    return sorted(name for name, pattern in _skill_patterns() if pattern.search(normalized))


def preprocess(text: str, return_stages: bool = False):
    normalized = normalize_text(text)
    # Canonical placeholders preserve C++, C#, .NET and phrase synonyms in TF-IDF.
    canonical = normalized
    for name, pattern in sorted(_skill_patterns(), key=lambda item: len(item[0]), reverse=True):
        placeholder = 'skill_' + re.sub(r'[^a-z0-9]', lambda m: '_' + str(ord(m[0])) + '_', name.casefold())
        canonical = pattern.sub(placeholder, canonical)
    tokens = re.findall(r'\b[^\W\d][\w]*\b', canonical, flags=re.UNICODE)
    filtered = [token for token in tokens if token not in ENGLISH_STOP_WORDS and token not in DOMAIN_STOP_WORDS]
    if return_stages:
        return filtered, {'1. Normalized': normalized, '2. Canonical skill aliases': canonical, '3. Tokenized': tokens, '4. Stopwords removed': filtered}
    return filtered


def _percent(value):
    value = float(value)
    return round(max(0.0, min(100.0, value)), 2) if math.isfinite(value) else 0.0


def tfidf_similarity(resume_text: str, job_description: str) -> float:
    docs = [preprocess(resume_text), preprocess(job_description)]
    if not all(docs):
        return 0.0
    vectors = TfidfVectorizer(analyzer=lambda tokens: tokens).fit_transform(docs)
    return _percent(cosine_similarity(vectors[0], vectors[1])[0, 0] * 100)


@lru_cache(maxsize=2)
def _load_semantic_model(model_name):
    # Cache failure too: missing optional packages/models must not delay each click.
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, local_files_only=True, device='cpu')
    except Exception:
        return None


def _semantic_similarity(resume_text, job_description):
    if os.getenv('RESUME_MATCHER_SEMANTIC', '').lower() not in {'1', 'true', 'yes'}:
        return None
    model = _load_semantic_model(os.getenv('RESUME_MATCHER_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    if model is None:
        return None
    try:
        # Mean-pool bounded chunks so long resumes are not silently reduced to their header.
        import numpy as np
        documents = []
        for text in (resume_text, job_description):
            words = text.split()
            chunks = [' '.join(words[i:i + 180]) for i in range(0, len(words), 180)]
            vectors = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            documents.append(np.mean(vectors, axis=0))
        value = float(cosine_similarity([documents[0]], [documents[1]])[0, 0])
        return _percent(value * 100) if math.isfinite(value) else None
    except Exception:
        return None


def analyze_resume(resume_text: str, job_description: str) -> dict:
    """Return JSON-compatible scores and gaps. Empty/stopword-only input scores zero.

    Without semantic inference: overall = 60% TF-IDF + 40% skill coverage.
    With semantic inference: overall = 35% TF-IDF + 25% semantic + 40% skills.
    If the JD has no recognized skills, remaining weights are renormalized.
    semantic_score=0 with semantic_available=False means unavailable, not a
    measured zero. Additional metadata lets frontends label fallback honestly.
    """
    resume, jd = normalize_text(resume_text), normalize_text(job_description)
    required = set(extract_skills(jd))
    present = set(extract_skills(resume))
    matched, missing = sorted(required & present), sorted(required - present)
    result = dict(overall_score=0.0, tfidf_score=0.0, semantic_score=0.0,
                  skill_score=0.0, matched_skills=matched, missing_skills=missing,
                  suggestions=[], semantic_available=False, scoring_method='tfidf',
                  skill_coverage_available=bool(required))
    if not preprocess(resume) or not preprocess(jd):
        result['suggestions'] = ['Provide both a readable resume and a job description containing meaningful words.']
        return result
    lexical = tfidf_similarity(resume, jd)
    semantic = _semantic_similarity(resume, jd)
    skill = _percent(100 * len(matched) / len(required)) if required else 0.0
    text_score = lexical if semantic is None else (0.35 * lexical + 0.25 * semantic) / 0.60
    overall = 0.60 * text_score + 0.40 * skill if required else text_score
    suggestions = [f'If you have experience with {name}, add a specific project, task, or measurable outcome demonstrating it; otherwise consider a small practice project.' for name in missing]
    if not required:
        suggestions.append('No dictionary skills were detected in the job description. Review its responsibilities manually; the overall score uses text similarity only.')
    elif not missing:
        suggestions.append('All detected job-description skills are mentioned. Strengthen these claims with specific projects and measurable outcomes.')
    if lexical < 40:
        suggestions.append('Tailor your summary and project descriptions to the actual responsibilities in this job description, using accurate terminology.')
    result.update(overall_score=_percent(overall), tfidf_score=lexical,
                  semantic_score=semantic if semantic is not None else 0.0,
                  skill_score=skill, suggestions=suggestions,
                  semantic_available=semantic is not None,
                  scoring_method='tfidf+semantic' if semantic is not None else 'tfidf')
    return result


def extract_pdf_text(source) -> str:
    """Read PDF bytes, a binary file, or a path; raise actionable ValueError on failure.

    Text PDFs only: scanned images require OCR outside this application.
    """
    try:
        from pypdf import PdfReader
        if isinstance(source, (bytes, bytearray)):
            source = BytesIO(source)
        if hasattr(source, 'seek'):
            source.seek(0)
        reader = PdfReader(source)
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('This PDF is password protected. Upload an unlocked copy.')
        if len(reader.pages) > 100:
            raise ValueError('Please upload a PDF with no more than 100 pages.')
        text = '\n'.join(page.extract_text() or '' for page in reader.pages).strip()
        if not text:
            raise ValueError('No readable text was found. Use a text-based PDF or paste the text; scanned PDFs require OCR.')
        normalize_text(text)  # Enforce the same document size limit as the analysis API.
        return text
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Unable to read this PDF. Upload a valid, unlocked text-based PDF or paste its text.') from exc


def read_uploaded_file(uploaded_file) -> str:
    """Streamlit-compatible upload helper for PDF and UTF-8 text."""
    name = getattr(uploaded_file, 'name', '').lower()
    if name.endswith('.pdf'):
        return extract_pdf_text(uploaded_file)
    if not name.endswith('.txt'):
        raise ValueError('Please upload a .pdf or .txt file.')
    uploaded_file.seek(0)
    try:
        text = uploaded_file.read().decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Please save the text file with UTF-8 encoding.') from exc
    normalize_text(text)
    return text
