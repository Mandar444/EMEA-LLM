from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.core.database import get_db
from backend.db.models import ChatSession, ChatMessage, DocumentChunk, Document
from backend.db.schemas import ChatMessageCreate, ChatMessageResponse, ChatSessionResponse, SourceInfo
from backend.ml.inference import inference_coordinator

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/session", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(db: Session = Depends(get_db)):
    """
    Creates a new conversation session.
    """
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_sessions(db: Session = Depends(get_db)):
    """
    Retrieves all conversation sessions.
    """
    return db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """
    Deletes a conversation session and all its messages.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )
    # Delete messages belonging to session
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    db.delete(session)
    db.commit()
    return None


@router.post("/query", response_model=ChatMessageResponse)
def query_assistant(payload: ChatMessageCreate, db: Session = Depends(get_db)):
    """
    Submits a query to the assistant.
    Retrieves document chunks, coordinates the hybrid BM25 + Semantic search,
    persists query and response messages, and outputs the matched passage with references.
    """
    # 1. Verify session exists
    session = db.query(ChatSession).filter(ChatSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation session not found."
        )

    # 2. Fetch all document chunks and document metadata from SQL
    # We join with the Document table so we have the filename of the source document
    chunks_query = db.query(
        DocumentChunk.id,
        DocumentChunk.document_id,
        DocumentChunk.chunk_index,
        DocumentChunk.content,
        DocumentChunk.vector_embedding,
        Document.filename
    ).join(Document, Document.id == DocumentChunk.document_id).all()

    # Convert query objects to dictionary format expected by InferenceCoordinator
    database_chunks = []
    for chunk in chunks_query:
        database_chunks.append({
            "id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "vector_embedding": chunk.vector_embedding,
            "filename": chunk.filename
        })

    # 3. Call Hybrid Retrieval Search
    response_text, confidence, source_info, source_type = inference_coordinator.search(
        query_text=payload.text,
        database_chunks=database_chunks,
        threshold=payload.confidence_threshold,
        alpha=payload.retrieval_alpha
    )

    # 4. Save User Message to Database
    user_msg = ChatMessage(
        session_id=payload.session_id,
        sender="user",
        text=payload.text
    )
    db.add(user_msg)
    
    # 5. Save System Message to Database
    system_msg = ChatMessage(
        session_id=payload.session_id,
        sender="system",
        text=response_text,
        retrieved_chunk_id=source_info["chunk_id"] if source_info else None,
        confidence_score=confidence
    )
    db.add(system_msg)
    db.commit()
    db.refresh(system_msg)

    # 6. Format Response Schema
    source_resp = None
    if source_info:
        contrib = source_info.get("contributing_sources")
        contrib_list = None
        if contrib:
            contrib_list = [
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "filename": c["filename"],
                    "chunk_index": c["chunk_index"],
                    "citation_index": c["citation_index"]
                }
                for c in contrib
            ]
        source_resp = SourceInfo(
            chunk_id=source_info["chunk_id"],
            document_id=source_info["document_id"],
            filename=source_info["filename"],
            chunk_index=source_info["chunk_index"],
            contributing_sources=contrib_list
        )

    return ChatMessageResponse(
        id=system_msg.id,
        session_id=system_msg.session_id,
        sender=system_msg.sender,
        text=system_msg.text,
        timestamp=system_msg.timestamp,
        confidence_score=system_msg.confidence_score,
        source=source_resp
    )


@router.get("/session/{session_id}/messages", response_model=List[ChatMessageResponse])
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """
    Fetches the chronological message history for a specific chat session.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )

    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
    
    response = []
    for msg in messages:
        source_resp = None
        if msg.retrieved_chunk:
            source_resp = SourceInfo(
                chunk_id=msg.retrieved_chunk.id,
                document_id=msg.retrieved_chunk.document_id,
                filename=msg.retrieved_chunk.document.filename,
                chunk_index=msg.retrieved_chunk.chunk_index
            )
            
        response.append(
            ChatMessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                sender=msg.sender,
                text=msg.text,
                timestamp=msg.timestamp,
                confidence_score=msg.confidence_score,
                source=source_resp
            )
        )
        
    return response
