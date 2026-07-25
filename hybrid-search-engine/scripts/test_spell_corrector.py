"""
Smoke test for SpellCorrector.
Usage: python scripts/test_spell_corrector.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.query.spell_check import SpellCorrector


def main() -> None:
    print("Initializing SpellCorrector...")
    corrector = SpellCorrector()

    try:
        corrector.load_default_dictionary()
        print("✓ Dictionary loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load dictionary: {e}\n")
        return

    test_queries = [
        "pythn frmwor",
        "machne lerning",
        "deep lerning modl",
        "how to optimze performance",
        "python framework",
        "exampel query",
    ]

    print("Testing spell correction:")
    print("-" * 60)

    for query in test_queries:
        try:
            corrected = corrector.correct_query(query)
            if corrected != query:
                print(f"✓ '{query}' → '{corrected}'")
            else:
                print(f"  '{query}' (no correction needed)")
        except Exception as e:
            print(f"✗ Error correcting '{query}': {e}")

    print("-" * 60)


if __name__ == "__main__":
    main()
