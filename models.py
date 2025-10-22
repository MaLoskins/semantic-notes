from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, ARRAY, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="notes")
    embedding = relationship("Embedding", back_populates="note", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id", ondelete="CASCADE"), unique=True, nullable=False)
    content_hash = Column(String(64), nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)
    model_name = Column(String(100), default="sentence-transformers/all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=datetime.utcnow)

    note = relationship("Note", back_populates="embedding")