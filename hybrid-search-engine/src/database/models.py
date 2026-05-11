"""
SQLAlchemy ORM models for the search engine database.
"""

import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Document(Base):
    """A searchable document."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    url = Column(String(2000), nullable=True)
    category = Column(String(100), nullable=True, index=True)
    view_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = Column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("idx_documents_category", "category"),
        Index("idx_documents_created_at", "created_at"),
    )


class SearchEvent(Base):
    """Records every search query and which results were shown."""

    __tablename__ = "search_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    query = Column(Text, nullable=False)
    corrected_query = Column(Text, nullable=True)
    result_doc_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    result_scores = Column(ARRAY(Float), nullable=False)
    latency_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ClickEvent(Base):
    """Records when a user clicks on a search result."""

    __tablename__ = "click_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    query = Column(Text, nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    position = Column(Integer, nullable=False)
    dwell_time_seconds = Column(Float, nullable=True)
    is_last_click = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_click_query_doc", "query", "document_id"),
        Index("idx_click_session", "session_id", "timestamp"),
    )


class ModelVersion(Base):
    """Tracks trained model versions and their performance metrics."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)
    version = Column(String(50), nullable=False)
    ndcg_at_10 = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)
    trained_on_samples = Column(Integer, nullable=True)
    file_path = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
