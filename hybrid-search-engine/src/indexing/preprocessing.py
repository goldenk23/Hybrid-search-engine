"""
Text preprocessing utilities for indexing and search result display.

The goal is not to change the meaning of text. We only remove structural
noise so BM25 indexes cleaner content.
"""

import html
import re


def clean_text(text: str | None) -> str:
    """
    Clean raw text before indexing.

    Steps:
    1. Convert None to empty string.
    2. Decode HTML entities like &amp; into &.
    3. Remove HTML tags.
    4. Collapse repeated whitespace.
    5. Strip leading and trailing whitespace.
    """
    if text is None:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_valid_document(
    title: str | None,
    body: str | None,
    min_body_length: int = 20,
    *,
    require_title: bool = True,
) -> bool:
    """Check if a document has enough useful content for indexing.

    Args:
        require_title: When True (default), a non-empty title is mandatory.
            Set to False for title-less corpora such as MS MARCO passages,
            where the collection has only an id and a body. The body length
            check always applies — it is the real signal of useful content.
    """
    if require_title and (not title or not title.strip()):
        return False

    return not (not body or len(body.strip()) < min_body_length)


def generate_snippet(body: str, query: str, snippet_length: int = 200) -> str:
    """Generate a short preview around the first query term found in the body."""
    if not body:
        return ""

    query_terms = query.lower().split()
    body_lower = body.lower()

    best_position = 0
    for term in query_terms:
        position = body_lower.find(term)
        if position != -1:
            best_position = position
            break

    start = max(0, best_position - snippet_length // 4)
    end = min(len(body), start + snippet_length)

    snippet = body[start:end].strip()

    if start > 0:
        snippet = "..." + snippet

    if end < len(body):
        snippet = snippet + "..."

    return snippet
