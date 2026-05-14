""" 
usage:python scripts/test_vector.py

"""

import shutil
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.search.vector import VectorSearch


temp_dir = Path(tempfile.gettempdir()) / "vector_search_test"

if temp_dir.exists():
    shutil.rmtree(temp_dir)

temp_dir.mkdir(parents=True, exist_ok=True)

documents = [
    {
        "id": "1",
        "title": "Python Programming and Automation",
        "body": """
Python is a high-level programming language widely used for backend development,
automation, scripting, APIs, machine learning, artificial intelligence, and data science.

Developers use Python for web frameworks like Django and FastAPI, data analysis with pandas,
and automation tasks such as file processing and web scraping.

Python is known for its simple syntax, readability, and huge ecosystem of libraries.
        """,
    },
    {
        "id": "2",
        "title": "PostgreSQL Relational Database",
        "body": """
PostgreSQL is an advanced open-source relational database management system.

It stores structured data in tables and supports SQL queries, indexing, transactions,
joins, constraints, and ACID compliance.

PostgreSQL is commonly used in backend systems, enterprise applications,
analytics platforms, and scalable web applications for reliable data storage.
        """,
    },
    {
        "id": "3",
        "title": "Stomach Pain and Abdominal Treatment",
        "body": """
Abdominal pain or stomach discomfort can occur because of digestion problems,
food poisoning, gastric infection, acidity, constipation, or ulcers.

Doctors may recommend hydration, proper diet, antacids, probiotics,
or medical treatment depending on the cause of the abdominal pain.

Symptoms may include cramps, nausea, bloating, vomiting, and digestive discomfort.
        """,
    },
]

engine = VectorSearch(index_path=temp_dir / "vector.faiss")
engine.doc_ids_path = temp_dir / "vector_doc_ids.npy"

engine.build_index(documents)

results = engine.search("how to fix abdominal pain", top_k=3)

for result in results:
    print(result)