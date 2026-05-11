"""
Initialize database tables for the hybrid search engine.

Usage:
    python scripts/init_db.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.postgres import init_db


if __name__ == "__main__":
    print("Initializing database tables...")
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as exc:
        print(f"Error initializing database: {exc}")
        sys.exit(1)
