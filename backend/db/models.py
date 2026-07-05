import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import relationship
from backend.core.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Document(Base):
    """
    Represents an uploaded enterprise document.
    """
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    uploaded_at = Column(DateTime, server_default=func.now())
    file_size = Column(Integer, nullable=False)

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """
    Represents a segmented text block (chunk) extracted from a Document.
    """
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # Store embedding vector as a JSON string for simple SQLite offline storage
    vector_embedding = Column(Text, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    """
    Represents an interactive conversation session.
    """
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """
    Represents a single message inside a ChatSession (User query or Assistant response).
    """
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    sender = Column(String(20), nullable=False)  # 'user' or 'system'
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    
    # Matching metadata for debugging & reference verification
    retrieved_chunk_id = Column(String(36), ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    retrieved_chunk = relationship("DocumentChunk")
