# 🎯 Git Commit Best Practices Guide

## Quick Reference

```bash
# Basic commit
git add <files>
git commit -m "type(scope): subject"

# View what you're committing
git diff --staged
git status

# View commit history
git log --oneline
git log --graph --oneline --all
```

---

## 1. Commit Message Format (Conventional Commits)

### Structure
```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Type (Required)
| Type | Usage | Example |
|------|-------|---------|
| `feat` | New feature | `feat: add BM25 scoring` |
| `fix` | Bug fix | `fix: resolve connection pool leak` |
| `docs` | Documentation | `docs: add API endpoint examples` |
| `refactor` | Code restructuring (no behavior change) | `refactor: simplify query builder` |
| `perf` | Performance improvement | `perf: optimize embedding indexing` |
| `test` | Add/modify tests | `test: add integration tests for search` |
| `chore` | Dependencies, config, build | `chore: update sqlalchemy version` |
| `ci` | CI/CD pipeline changes | `ci: add GitHub Actions workflow` |
| `style` | Formatting, missing semicolons | `style: format models.py with black` |

### Scope (Optional but Recommended)
Specify which part of the system is affected:
- `database` - ORM models, migrations
- `search` - Search algorithms, ranking
- `api` - REST endpoints, handlers
- `config` - Configuration files
- `docker` - Docker setup
- `scripts` - Utility scripts

### Subject (Required)
- ✅ Imperative mood: "add" not "added" or "adds"
- ✅ Don't capitalize first letter
- ✅ No period at the end
- ✅ Max 50 characters
- ❌ Don't say "fix bug" - say what the bug was

### Body (Optional but Recommended for Complex Changes)
- Explain **WHY** the change was needed
- Explain **WHAT** changed
- Explain **HOW** it works if non-obvious
- Wrap at 72 characters
- Use bullet points for clarity

### Footer (Optional)
For referencing issues or breaking changes:
```
Closes #123
Breaking-change: old API endpoint /search/v1 is deprecated
```

---

## 2. Full Commit Message Examples

### ✅ GOOD: Simple Feature
```
feat(database): add search event tracking table

SearchEvent table records query impressions and latencies
for training the LTR ranker.
```

### ✅ GOOD: Bug Fix with Context
```
fix(search): handle empty query strings gracefully

Previously, empty queries crashed the BM25 tokenizer.
Now we return a user-friendly error message.

Closes #42
```

### ✅ GOOD: Complex Feature with Details
```
feat(search): implement hybrid search combining BM25 and embeddings

Adds hybrid search ranking that combines:
- BM25 lexical matching (80% weight)
- Dense embedding similarity (20% weight)
- Query normalization for consistency

Ranking formula: final_score = 0.8 * bm25_score + 0.2 * embedding_score

Performance: ~200ms for 10K documents with <5% accuracy improvement

Closes #18
```

### ✅ GOOD: Refactoring
```
refactor(api): extract query validation into separate module

Moves validation logic from search_handler.py to query_validator.py
for reusability and testability. No functional changes.

Tests updated to reflect new module structure.
```

### ❌ BAD: Vague
```
fix: stuff
update: changed things
wip: work in progress
fixed bugs
```

### ❌ BAD: Too Long and Unclear
```
feat: added new stuff and fixed some issues and updated dependencies and made changes to database
```

---

## 3. Pre-Commit Checklist

Before running `git commit`, verify:

```bash
# Step 1: Check git status
git status

# Step 2: Review changes before staging
git diff

# Step 3: Stage only relevant files
git add src/database/models.py
# OR add everything if all changes are for one commit
git add .

# Step 4: Review staged changes (should match your intent)
git diff --staged

# Step 5: Run linting/formatting (if applicable)
black src/
flake8 src/
# pylint src/

# Step 6: Run tests (if applicable)
pytest tests/

# Step 7: Commit with message
git commit -m "type(scope): message"

# Step 8: Verify commit
git log --oneline -1
git show HEAD
```

---

## 4. Atomic Commits (Golden Rule)

**Principle:** One logical change per commit

### ✅ GOOD: Atomic Commits
```
Commit 1: feat(database): add Document model
Commit 2: feat(database): add SearchEvent model
Commit 3: feat(database): add indexes for performance
Commit 4: test(database): add model unit tests
```

### ❌ BAD: Mixed Concerns
```
Commit 1: feat: added models, updated dependencies, fixed bug in API, reformatted code
```

**Why atomic matters:**
- Easy to review (one thing at a time)
- Easy to revert if broken (`git revert <commit>`)
- Easy to bisect and find bugs (`git bisect`)
- Clear project history

---

## 5. Commit Workflow for This Project

### For New Features
```bash
git checkout -b feature/hybrid-search
# ... make changes ...
git add src/search/hybrid.py
git commit -m "feat(search): implement hybrid BM25+embedding ranking"
git push origin feature/hybrid-search
# Create Pull Request
```

### For Bug Fixes
```bash
git checkout -b fix/query-parsing
# ... fix the bug ...
git add src/search/parser.py tests/test_parser.py
git commit -m "fix(search): handle special characters in queries"
git push origin fix/query-parsing
# Create Pull Request
```

### For Documentation
```bash
git add docs/API.md README.md
git commit -m "docs: add hybrid search architecture overview"
git push
```

---

## 6. Viewing Commit History

```bash
# Simple log
git log --oneline

# With graph (shows branches)
git log --graph --oneline --all

# With details
git log -p

# Last N commits
git log -n 5

# Search for keyword
git log --grep="database"

# By author
git log --author="your-name"

# Since/until dates
git log --since="2024-01-01" --until="2024-12-31"
```

---

## 7. Fixing Commits

### Mistake: Typo in commit message
```bash
git commit --amend -m "feat(database): corrected message"
```

### Mistake: Forgot to stage a file
```bash
git add forgotten_file.py
git commit --amend --no-edit
```

### Mistake: Committed to wrong branch
```bash
git log --oneline  # find your commit hash
git reset HEAD~1   # undo the commit, keep changes
git checkout correct-branch
git add .
git commit -m "correct message"
```

⚠️ **WARNING:** Don't amend commits already pushed to shared branches!

---

## 8. Common Mistakes to Avoid

| ❌ Mistake | ✅ Fix | Why |
|-----------|--------|-----|
| Committing `.pyc`, `__pycache__`, `.env` | Add to `.gitignore` | These are artifacts, not source code |
| Committing `print()` debugging statements | Remove before commit | Clutters production logs |
| Committing large files (>100MB) | Use Git LFS or remove | Bloats repository |
| Committing secrets (API keys, passwords) | Use `.env` and `.gitignore` | Security risk |
| Vague messages like "wip", "fix", "update" | Use type(scope): format | Unreadable history |
| Mixed concerns in one commit | Create separate commits | Hard to review, hard to revert |
| Uncommitted changes left in working directory | `git status` before pushing | Can cause confusion |
| Force pushing to `main` branch | Never do this | Breaks team collaboration |
| Merging without understanding conflicts | Read conflict markers carefully | Can break code |

---

## 9. .gitignore Template for Python

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
data/raw/
models/checkpoints/
*.pkl
```

---

## 10. Quick Command Reference

```bash
# View differences
git diff                    # unstaged changes
git diff --staged           # staged changes only
git diff HEAD~1             # compare to last commit

# Staging/Unstaging
git add <file>              # stage file
git add .                   # stage all
git add -p                  # stage interactively (choose hunks)
git reset <file>            # unstage file
git restore <file>          # discard changes to file

# Committing
git commit -m "message"     # create commit
git commit -am "message"    # stage tracked files + commit
git commit --amend          # modify last commit

# Viewing history
git log                     # full log
git log --oneline           # short format
git log --oneline -n 10     # last 10 commits
git show <commit>           # show commit details

# Undoing
git revert <commit>         # create new commit that undoes changes
git reset --soft HEAD~1     # undo commit, keep changes staged
git reset --hard HEAD~1     # undo commit, discard changes ⚠️
```

---

## 11. For This Project: Example Commits

### Commit 1 (Models)
```
feat(database): add ORM models for search engine

Adds four core SQLAlchemy models:
- Document: searchable passages with metadata
- SearchEvent: query impressions and latency tracking
- ClickEvent: user interactions with click signals
- ModelVersion: LTR model versioning

All models include proper indexing and comprehensive docstrings.
```

### Commit 2 (Scripts)
```
feat(scripts): add MS MARCO dataset downloader

Script downloads MS MARCO passages and qrels files.
Supports filtering by dataset size (small, medium, full).

Usage: python scripts/download_msmarco.py --size small
```

### Commit 3 (Config)
```
feat(config): add environment configuration module

Centralizes all configuration:
- Database connection strings
- Model paths and hyperparameters
- Search engine settings

Loads from environment variables with sensible defaults.
```

---

## 12. Key Takeaways

| Principle | Remember |
|-----------|----------|
| **Atomic** | One logical change per commit |
| **Clear** | Use type(scope): format |
| **Descriptive** | Explain WHY, not just WHAT |
| **Testable** | Each commit should be deployable |
| **Reviewable** | Keep commits small enough to review in <5min |
| **Searchable** | Future you will grep the history |

---

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Atomic Commits Best Practices](https://www.freshcodeblock.com/blog/atomic-commits/)

---

**Happy committing! 🚀**

Remember: Good commits today = easy debugging tomorrow.
