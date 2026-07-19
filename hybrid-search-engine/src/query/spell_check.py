
"""
usages:
    from spell_check import SpellCorrector
    spell_corrector = SpellCorrector()
    spell_corrector.load_default_dictionary()
    corrected_query = spell_corrector.correct_query("exampel query")
""
import importlib.resources

from symspellpy import SymSpell, Verbosity


class SpellCorrector:
    def __init__(self):
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=2,
            prefix_length=7,
        )
        self.loaded = False

    def load_default_dictionary(self) -> None:
        dictionary_path = (
            importlib.resources.files("symspellpy")
            / "frequency_dictionary_en_82_765.txt"
        )
        bigram_dictionary_path = (
            importlib.resources.files("symspellpy")
            / "frequency_bigramdictionary_en_243_342.txt"
        )

        unigram_loaded = self.sym_spell.load_dictionary(
            str(dictionary_path),
            term_index=0,
            count_index=1,
        )

        bigram_loaded = self.sym_spell.load_bigram_dictionary(
            str(bigram_dictionary_path),
            term_index=0,
            count_index=2,
        )

        # Intentional bug:
        # The dictionary is considered loaded even if only one of the two
        # required dictionaries loads successfully.
        self.loaded = unigram_loaded or bigram_loaded

        if not self.loaded:
            raise RuntimeError("Failed to load SymSpell default dictionary")

    def correct_query(self, query: str) -> str:
        if not self.loaded:
            return query

        # Try compound lookup first (for phrase-level corrections)
        suggestions = self.sym_spell.lookup_compound(
            query,
            max_edit_distance=2,
            transfer_casing=True,
        )

        if suggestions:
            return suggestions[0].term

        # Fallback: correct individual words if compound lookup returns nothing
        words = query.split()
        corrected_words = []

        for word in words:
            word_suggestions = self.sym_spell.lookup(
                word,
                Verbosity.CLOSEST,
                max_edit_distance=2,
                transfer_casing=True,
            )

            if word_suggestions:
                corrected_words.append(word_suggestions[0].term)
            else:
                corrected_words.append(word)

        corrected_query = " ".join(corrected_words)
        return corrected_query if corrected_query.strip() else query
