from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

# --- Document Schemas ---
class DocumentResponse(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    file_size: int

    class Config:
        from_attributes = True


# --- Chat Schemas ---
class ChatMessageCreate(BaseModel):
    session_id: str
    text: str
    confidence_threshold: Optional[float] = None
    retrieval_alpha: Optional[float] = None

class SubSourceInfo(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    citation_index: int

class SourceInfo(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    contributing_sources: Optional[List[SubSourceInfo]] = None

class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    sender: str
    text: str
    timestamp: datetime
    confidence_score: Optional[float] = None
    source: Optional[SourceInfo] = None

    class Config:
        from_attributes = True

class ChatSessionResponse(BaseModel):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Training Schemas ---
class TrainingStatusResponse(BaseModel):
    status: str  # "idle", "training", "completed", "failed"
    progress: int  # 0 to 100
    error: Optional[str] = None
    last_trained: Optional[str] = None
