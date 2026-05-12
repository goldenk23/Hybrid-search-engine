# tests/test_preprocessing.py
from src.indexing.preprocessing import clean_text, generate_snippet, is_valid_document


# assert (condition)
# assert checks whether a condition is True.
# If the condition is True -> test passes silently.
# If the condition is False -> Python raises an AssertionError and the test fails.
def test_clean_text_removes_html_and_decodes_entities():
    raw = "<p>Python &amp; FastAPI</p>\n\nBuild APIs     quickly."
    cleaned = clean_text(raw)

    assert cleaned == "Python & FastAPI Build APIs quickly."


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_is_valid_document_rejects_empty_title():
    assert is_valid_document("", "This body is long enough to be valid.") is False


def test_is_valid_document_rejects_short_body():
    assert is_valid_document("Title", "too short") is False


def test_generate_snippet_contains_query_term():
    body = "Python is useful for APIs, automation, data science, and search systems."
    snippet = generate_snippet(body, "automation", snippet_length=50)

    assert "automation" in snippet.lower()
