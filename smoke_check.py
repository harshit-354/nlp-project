"""
smoke_check.py
==============
Automated smoke test verifying:
1. analysis_api contract adherence (keys, types, bounds)
2. app.py module exports and compatibility
3. Streamlit AppTest end-to-end simulation
"""

import sys
import os

def test_analysis_api_contract():
    print("[1/3] Testing analysis_api contract...")
    import analysis_api

    assert hasattr(analysis_api, "analyze_resume"), "analysis_api missing analyze_resume"

    with open("samples/resume.txt") as f:
        resume = f.read()
    with open("samples/job_description.txt") as f:
        jd = f.read()

    result = analysis_api.analyze_resume(resume, jd)
    assert isinstance(result, dict), "Result must be a dict"

    expected_keys = [
        "overall_score",
        "tfidf_score",
        "semantic_score",
        "skill_score",
        "matched_skills",
        "missing_skills",
        "suggestions",
    ]
    for key in expected_keys:
        assert key in result, f"Missing required key: {key}"

    # Bounds & types
    assert 0 <= result["overall_score"] <= 100, f"overall_score out of bounds: {result['overall_score']}"
    assert 0 <= result["tfidf_score"] <= 100, f"tfidf_score out of bounds: {result['tfidf_score']}"
    if result["semantic_score"] is not None:
        assert 0 <= result["semantic_score"] <= 100, f"semantic_score out of bounds: {result['semantic_score']}"
    assert 0 <= result["skill_score"] <= 100, f"skill_score out of bounds: {result['skill_score']}"

    assert isinstance(result["matched_skills"], list), "matched_skills must be a list"
    assert isinstance(result["missing_skills"], list), "missing_skills must be a list"
    assert isinstance(result["suggestions"], list), "suggestions must be a list"

    # Edge cases
    empty_result = analysis_api.analyze_resume("", "")
    for key in expected_keys:
        assert key in empty_result, f"Missing key in empty input result: {key}"

    print("  -> analysis_api contract PASSED!")


def test_app_backward_compatibility():
    print("[2/3] Testing app.py exports and backward compatibility...")
    import app

    expected_callables = [
        "preprocess",
        "vectorize",
        "similarity",
        "keyword_gap",
        "semantic_similarity",
        "read_uploaded_file",
        "render_tags",
        "main",
    ]
    for fn in expected_callables:
        assert hasattr(app, fn), f"app.py missing function: {fn}"
        assert callable(getattr(app, fn)), f"{fn} in app.py is not callable"

    # Test preprocess return_stages
    tokens, stages = app.preprocess("Python machine learning engineer 2026", return_stages=True)
    assert isinstance(tokens, list), "preprocess tokens must be a list"
    assert isinstance(stages, dict), "preprocess stages must be a dict"
    assert len(stages) == 6, "Expected 6 preprocessing stages"

    print("  -> app.py backward compatibility PASSED!")


def test_streamlit_apptest_e2e():
    print("[3/3] Testing Streamlit AppTest end-to-end execution...")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run()
    assert len(at.exception) == 0, f"Initial run exceptions: {at.exception}"

    # Click Load Demo
    demo_buttons = [b for b in at.button if "Load Demo" in b.label]
    assert len(demo_buttons) > 0, "Load Demo button not found"
    demo_buttons[0].click().run()

    assert len(at.exception) == 0, f"Exceptions after Load Demo: {at.exception}"
    print("  -> Streamlit AppTest execution PASSED!")


if __name__ == "__main__":
    test_analysis_api_contract()
    test_app_backward_compatibility()
    test_streamlit_apptest_e2e()
    print("\nALL SMOKE CHECKS PASSED SUCCESSFULLY! Ready for demo.")
