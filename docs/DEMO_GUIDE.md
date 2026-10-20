# Demonstration Guide

Install `requirements.txt` once in an activated Python 3.10+ environment, then
launch with `python -m streamlit run app.py`. Open http://localhost:8501.
The basic demonstration needs no internet, API key or semantic model.

1. Select **Junior NLP (Priya)** under **Quick Demo**, then **Load Demo**.
   Explain that these are illustrative inputs and all displayed scores are
   calculated from their actual text.
2. Point to the circular gauge and separate TF-IDF / skill coverage cards.
   Semantic is unavailable unless a local model is prepared and enabled.
3. Show the Altair score chart, matched/missing skills and skill filters.
   Explain dictionary aliases and that a mention does not prove expertise.
4. Open the NLP pipeline expander. Show the four actual preprocessing stages,
   jointly fitted TF-IDF vocabulary and cosine angle. There is no lemmatization.
5. Load **NLP Scientist (Jordan)**, **Data Engineer (Alex)** or **Content Strategy
   Comparison** to demonstrate different computed matches. No fixed ranking is
   promised; inspect the evidence behind each score.
6. Download JSON and Markdown reports. Unavailable scores are clearly labelled.
7. Paste `Python SQL` and `Python SQL Docker`, then analyze. Docker should be
   missing and skill coverage should be approximately 66.67%.
8. Explain offline weights (60% text, 40% skills), optional semantic weights
   (35% text, 25% semantic, 40% skills), and the limitations in the README.

Text-based PDFs and UTF-8 TXT files can be uploaded into the two input panels.
Scanned PDFs need OCR elsewhere. Inputs are limited to 100,000 characters and
PDFs to 100 pages. **Clear All** resets inputs and results. Editing inputs clears
stale results until the next analysis. See `docs/VIVA_NOTES.md` for detailed Q&A.
