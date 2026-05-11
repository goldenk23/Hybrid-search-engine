"""
PostgreSQL connection management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import ASYNC_POSTGRES_URL, POSTGRES_URL
from src.database.models import Base

engine = create_engine(
    POSTGRES_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(bind=engine)

async_engine = create_async_engine(
    ASYNC_POSTGRES_URL,
    pool_size=10,
    max_overflow=20,
)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession)


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def get_session() -> Session:
    """Get a synchronous database session."""
    return SessionLocal()


async def get_async_session() -> AsyncSession:
    """Get an async database session for FastAPI dependency injection."""
    async with AsyncSessionLocal() as session:
        yield session
