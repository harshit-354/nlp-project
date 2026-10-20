"""Offline API/PDF smoke checks: python -m unittest discover -s tests -v."""
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import analysis_api as api


def sample_pdf(text='Python SQL Docker'):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class AnalysisSmokeTests(unittest.TestCase):
    def test_sample_and_contract(self):
        root = Path(__file__).resolve().parents[1]
        result = api.analyze_resume((root / 'samples/resume.txt').read_text(), (root / 'samples/job_description.txt').read_text())
        for key in ('overall_score', 'tfidf_score', 'semantic_score', 'skill_score'):
            self.assertIsInstance(result[key], float)
            self.assertTrue(0 <= result[key] <= 100)
        for key in ('matched_skills', 'missing_skills', 'suggestions'):
            self.assertIsInstance(result[key], list)
            self.assertTrue(all(isinstance(value, str) for value in result[key]))
        self.assertGreater(result['overall_score'], 0)

    def test_aliases_and_boundaries(self):
        result = api.analyze_resume('PYTHON3, sklearn, NLP, machine-learning, C++, postgres', 'Python scikit-learn natural language processing ML C++ PostgreSQL Docker')
        self.assertEqual(result['missing_skills'], ['Docker'])
        self.assertEqual(len(result['matched_skills']), 6)
        self.assertNotIn('Java', api.extract_skills('JavaScript'))
        self.assertNotIn('R', api.extract_skills('research'))
        self.assertIn('C#', api.extract_skills('C# and .NET'))
        self.assertIn('.NET', api.extract_skills('C# and .NET'))
        self.assertEqual(api.tfidf_similarity('NLP sklearn C++', 'natural language processing scikit-learn cpp'), 100.0)

    def test_empty_and_unusual_inputs(self):
        for resume, jd in [('', ''), ('Python', ''), ('', 'Python'), ('the and', 'Python'), ('!!!123', 'Python')]:
            result = api.analyze_resume(resume, jd)
            self.assertEqual(result['overall_score'], 0)
            self.assertTrue(result['suggestions'])
        self.assertEqual(api.analyze_resume('Ｐｙｔｈｏｎ', 'python')['overall_score'], 100)
        self.assertEqual(api.analyze_resume('你好 世界', '你好 世界')['tfidf_score'], 100)
        with self.assertRaises(TypeError):
            api.analyze_resume(None, 'Python')
        with self.assertRaises(ValueError):
            api.analyze_resume('x' * 100001, 'Python')

    def test_scores_and_fallback(self):
        with patch.dict(os.environ, {'RESUME_MATCHER_SEMANTIC': '1'}), patch.object(api, '_load_semantic_model', return_value=None):
            result = api.analyze_resume('Python', 'Python Docker')
        self.assertFalse(result['semantic_available'])
        self.assertEqual(result['semantic_score'], 0)
        self.assertEqual(result['skill_score'], 50)
        self.assertAlmostEqual(result['overall_score'], .6 * result['tfidf_score'] + 20, places=2)
        with patch.object(api, '_semantic_similarity', return_value=80):
            result = api.analyze_resume('Python', 'Python Docker')
        self.assertTrue(result['semantic_available'])
        self.assertAlmostEqual(result['overall_score'], .35 * result['tfidf_score'] + 20 + 20, places=2)
        result = api.analyze_resume('gardening flowers', 'gardening flowers')
        self.assertFalse(result['skill_coverage_available'])
        self.assertEqual(result['overall_score'], 100)

    def test_pdf_to_real_analysis(self):
        text = api.extract_pdf_text(sample_pdf())
        result = api.analyze_resume(text, 'Python SQL Docker Kubernetes')
        self.assertEqual(result['matched_skills'], ['Docker', 'Python', 'SQL'])
        self.assertEqual(result['missing_skills'], ['Kubernetes'])
        self.assertEqual(result['skill_score'], 75)
        with self.assertRaises(ValueError):
            api.extract_pdf_text(b'not a PDF')
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        output = io.BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(ValueError, 'No readable text'):
            api.extract_pdf_text(output.getvalue())
        writer.encrypt('secret')
        output = io.BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(ValueError, 'password protected'):
            api.extract_pdf_text(output.getvalue())

    def test_semantic_override_is_request_scoped(self):
        with patch.dict(os.environ, {'RESUME_MATCHER_SEMANTIC': '1'}), patch.object(api, '_load_semantic_model', return_value=None) as loader:
            result = api.analyze_resume('Python', 'Python', use_semantic=False)
            loader.assert_not_called()
            self.assertFalse(result['semantic_available'])
            self.assertEqual(os.environ['RESUME_MATCHER_SEMANTIC'], '1')
            api.analyze_resume('Python', 'Python', use_semantic=True)
            loader.assert_called_once()

    def test_semantic_failure_is_cached(self):
        api._load_semantic_model.cache_clear()
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            self.assertIsNone(api._load_semantic_model('missing-model'))
            self.assertIsNone(api._load_semantic_model('missing-model'))
        self.assertEqual(api._load_semantic_model.cache_info().hits, 1)


if __name__ == '__main__':
    unittest.main()
