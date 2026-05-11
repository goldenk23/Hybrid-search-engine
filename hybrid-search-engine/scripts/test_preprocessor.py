import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.preprocessing import clean_text, generate_snippet, is_valid_document

raw_text = """
<p>Python &amp; FastAPI</p>

FastAPI is a modern     web framework for building APIs.
"""

cleaned = clean_text(raw_text)

print("Cleaned:")
print(cleaned)

print("\nValid document?")
print(is_valid_document("FastAPI Tutorial", cleaned))

print("\nSnippet:")
print(generate_snippet(cleaned, "web framework", snippet_length=80))
