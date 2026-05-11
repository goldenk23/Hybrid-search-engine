# Quick Start & Development Guide

## 🚀 Quick Start Commands

### 1. Start Services (PostgreSQL + Redis)
```bash
docker compose up -d
```

### 2. Initialize Database (Create Tables)
```bash
python scripts/init_db.py
```

### 3. Check Container Status
```bash
docker compose ps
```

### 4. View Logs
```bash
# PostgreSQL logs
docker compose logs -f postgres

# Redis logs
docker compose logs -f redis

# All services logs
docker compose logs -f
```

### 5. Stop Services
```bash
docker compose down
```

---

## 📋 Full Setup Workflow

For a fresh setup, run these commands in order:

```bash
# 1. Navigate to project directory
cd hybrid-search-engine

# 2. Create virtual environment (if not already done)
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell

# 3. Install dependencies
pip install -e .

# 4. Start Docker services
docker compose up -d

# 5. Wait a few seconds for PostgreSQL to be ready
Start-Sleep -Seconds 10

# 6. Initialize database
python scripts/init_db.py

# 7. Verify everything is working
docker compose ps
```

---

## 🔧 Development Workflow

### Daily Development

```bash
# Start services
docker compose up -d

# Run your application
python -m uvicorn src.main:app --reload

# In another terminal, view logs
docker compose logs -f postgres
```

### Database Management

```bash
# Reset database (WARNING: deletes all data)
docker compose down -v           # Remove volumes
docker compose up -d             # Start fresh
python scripts/init_db.py        # Recreate tables

# Access PostgreSQL directly
docker exec -it search-postgres psql -U search_user -d search_engine

# Access Redis CLI
docker exec -it search-redis redis-cli
```

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py
```

---

## 🐳 Docker Commands Reference

```bash
# View running containers
docker compose ps

# View all containers (including stopped)
docker compose ps -a

# View container logs
docker compose logs <service>     # postgres, redis
docker compose logs -f <service>  # Follow logs in real-time

# Execute command inside container
docker exec <container_name> <command>

# Stop all services
docker compose stop

# Start services again
docker compose start

# Remove services and volumes
docker compose down           # Keep volumes
docker compose down -v        # Remove volumes (data loss!)

# Rebuild containers
docker compose build

# View resource usage
docker stats
```

---

## 📝 Environment Variables

By default, the `.env` file (git-ignored) is loaded automatically. Create one in the project root:

```bash
# .env
POSTGRES_URL=postgresql://search_user:search_password@localhost:5432/search_engine
REDIS_URL=redis://localhost:6379/0

# Search settings
BM25_TOP_K=100
VECTOR_TOP_K=100
RERANK_TOP_K=50
RESULTS_PER_PAGE=10

# Model paths
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Override any value by setting the environment variable before running:
```bash
$env:POSTGRES_URL="postgresql://user:pass@remote-host:5432/db"
python scripts/init_db.py
```

---

## ❌ Troubleshooting

### PostgreSQL Connection Refused
```bash
# Check if containers are running
docker compose ps

# Restart PostgreSQL
docker compose restart postgres

# Wait for it to be healthy
docker compose ps   # Check STATUS column

# View logs
docker compose logs postgres
```

### Port Already in Use
```bash
# Find what's using the port
netstat -ano | findstr :5432

# If it's a Docker container, stop it
docker stop <container_name>

# If it's another application, stop it or use different port
# Edit docker-compose.yml to use different port:
# ports:
#   - "5433:5432"  # Use 5433 instead
```

### Can't Connect to Database
```bash
# Verify credentials in .env match docker-compose.yml
# Test connection directly
docker exec -it search-postgres psql -U search_user -d search_engine -c "SELECT 1;"

# Check network connectivity
docker exec -it search-postgres ping redis
```

### Database Tables Not Created
```bash
# Check if init_db ran successfully
python scripts/init_db.py

# Connect to database and check tables
docker exec -it search-postgres psql -U search_user -d search_engine -c "\dt"
```

---

## 📚 Project Structure

```
hybrid-search-engine/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (env vars)
│   ├── database/
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   ├── postgres.py        # PostgreSQL connection pool
│   │   └── redis_client.py    # Redis cache client
│   └── main.py                # FastAPI application
├── scripts/
│   ├── init_db.py             # Initialize database tables
│   └── download_msmarco.py    # Download MS MARCO dataset
├── tests/
│   └── test_*.py              # Unit tests
├── data/
│   ├── msmarco/               # MS MARCO dataset
│   └── indexes/               # Search indexes
├── docker-compose.yml         # Docker services config
├── pyproject.toml             # Project metadata & dependencies
├── .env                       # Environment variables (git-ignored)
├── .gitignore                 # Git ignore rules
├── GIT_COMMIT_GUIDE.md        # Git commit best practices
└── QUICK_START.md             # This file
```

---

## ✅ Verification Checklist

After setup, verify everything:

- [ ] `docker compose ps` shows both `postgres` and `redis` as healthy
- [ ] `python scripts/init_db.py` runs without errors
- [ ] Can connect to PostgreSQL: `docker exec search-postgres pg_isready`
- [ ] Can connect to Redis: `docker exec search-redis redis-cli ping`
- [ ] Tables exist: `docker exec search-postgres psql -U search_user -d search_engine -c "\dt"`

---

## 🚀 Next Steps

1. Create API endpoints in `src/main.py`
2. Implement BM25 search in `src/search/bm25.py`
3. Implement vector search in `src/search/vector.py`
4. Add learning-to-rank model in `src/models/ltr.py`
5. Create integration tests in `tests/`

---

## 📞 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e .` |
| `Connection refused` | Wait longer or restart: `docker compose restart postgres` |
| `Port already in use` | Stop conflicting container or use different port |
| `Database locked` | Restart PostgreSQL container |
| `Redis connection error` | Check Redis is running: `docker compose ps` |

---

**Happy coding! 🎉**
