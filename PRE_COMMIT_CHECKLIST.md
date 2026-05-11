# Pre-Commit Checklist

Use this checklist before every commit to ensure quality and consistency.

## ✅ Code Quality

- [ ] Code follows project style (black, flake8, pylint)
- [ ] No debugging `print()` statements left
- [ ] No commented-out code blocks
- [ ] Type hints added where applicable
- [ ] Docstrings added for functions/classes

## ✅ Testing

- [ ] New code has unit tests
- [ ] Existing tests still pass: `pytest tests/`
- [ ] No hardcoded values or magic numbers
- [ ] Edge cases handled (empty strings, None values, etc.)

## ✅ Git Hygiene

- [ ] Reviewed changes with `git diff --staged`
- [ ] Only relevant files are staged
- [ ] No accidental file additions (check `.gitignore`)
- [ ] Commit message follows Conventional Commits format
- [ ] Commit message is clear and descriptive

## ✅ What NOT to Commit

- [ ] ❌ `.env` files or secrets
- [ ] ❌ `__pycache__/` or `.pyc` files
- [ ] ❌ `venv/` or virtual environment folders
- [ ] ❌ IDE-specific files (`.vscode/settings.json`, `.idea/`)
- [ ] ❌ Build artifacts or compiled files
- [ ] ❌ Large binary files (>100MB)
- [ ] ❌ Unrelated changes (keep commits atomic)

## ✅ Commit Message Checklist

- [ ] Type is correct (feat, fix, docs, refactor, test, chore, etc.)
- [ ] Scope is specified if applicable (database, search, api, etc.)
- [ ] Subject is in imperative mood ("add", not "added")
- [ ] Subject doesn't end with period
- [ ] Subject is under 50 characters
- [ ] Body explains WHY, not just WHAT
- [ ] Body lines wrapped at 72 characters
- [ ] Body is separated from subject by blank line

## Quick Commands

```bash
# Before committing
git status                    # Check what's staged
git diff --staged             # Review changes
black src/                    # Format code
flake8 src/                   # Check style
pytest tests/                 # Run tests

# Making the commit
git add <files>               # Stage files
git commit -m "type(scope): message"

# After committing
git log --oneline -1          # Verify commit
git show HEAD                 # Show commit details
```

## Example Pre-Commit Session

```bash
# 1. Review status
$ git status
On branch main
Untracked files:
  src/search/hybrid.py
  tests/test_hybrid.py

# 2. Review changes
$ git diff src/search/hybrid.py
# [review code...]

# 3. Format code
$ black src/search/hybrid.py

# 4. Run linting
$ flake8 src/search/hybrid.py

# 5. Run tests
$ pytest tests/test_hybrid.py
# [tests pass...]

# 6. Stage files
$ git add src/search/hybrid.py tests/test_hybrid.py

# 7. Review staged changes
$ git diff --staged
# [review one more time...]

# 8. Commit
$ git commit -m "feat(search): implement hybrid BM25+embedding ranking"

# 9. Verify
$ git log --oneline -1
a1b2c3d feat(search): implement hybrid BM25+embedding ranking
```

---

**Pro Tip:** Run this before each commit:

```bash
git diff --staged && git status
```

If the diff is small and the status is clean, you're ready to commit!
