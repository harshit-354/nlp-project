#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python - <<'PY'
import sys
from pathlib import Path
assert sys.version_info >= (3, 10), 'Python 3.10+ required'
import sklearn, pypdf, streamlit, altair, pandas
from analysis_api import analyze_resume
for name in ('datascientist_resume.txt', 'data_engineer_resume.txt', 'python_developer.txt', 'content_writer.txt'):
    assert Path('demo_data/samples', name).read_text().strip(), name
for name in ('DEMO_GUIDE.md', 'VIVA_NOTES.md'):
    assert Path('docs', name).read_text().strip(), name
result = analyze_resume(Path('samples/resume.txt').read_text(), Path('samples/job_description.txt').read_text(), use_semantic=False)
assert result['overall_score'] > 0 and result['matched_skills']
print('Dependency, documentation, sample and real analysis smoke checks PASSED')
print({key: result[key] for key in ('overall_score', 'tfidf_score', 'skill_score')})
PY
