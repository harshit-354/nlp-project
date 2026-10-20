# Viva Cheat Sheet

- **Pipeline:** Unicode normalization → skill aliases → tokenization → bundled
  stopword removal → TF-IDF and cosine → dictionary gaps → recommendations.
- **TF-IDF:** fitted on the resume and JD together, not on JD lines.
- **Coverage:** matched dictionary skills / detected JD skills × 100.
- **Overall:** 60% lexical + 40% skills offline; 35% lexical + 25% semantic +
  40% skills when available. No JD skills: renormalize text weights.
- **Reliability:** no runtime downloads, optional cached semantic model,
  malformed PDF handling, bounded inputs and automated tests.
- **Limits:** no OCR, no proficiency/negation inference, English-oriented finite
  dictionary; scores are not hiring probabilities.

Read [24 technical questions and answers](docs/VIVA_NOTES.md) and the
[presentation walkthrough](docs/DEMO_GUIDE.md).
