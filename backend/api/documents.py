from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from pathlib import Path
import os
import shutil

from backend.config import RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from backend.core.database import get_db
from backend.core.document_parser import process_document
from backend.db.models import Document, DocumentChunk
from backend.db.schemas import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads an enterprise document, cleans it, segments it into overlapping chunks,
    and stores both document metadata and chunks in the SQLite database.
    """
    # 1. Validate File Extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt", ".md", ".docx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only .pdf, .txt, .md, and .docx files are supported."
        )

    # 2. Save Document to raw storage
    file_path = RAW_DOCS_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document to storage: {str(e)}"
        )

    # 3. Create Document DB entry
    db_doc = Document(
        filename=file.filename,
        file_path=str(file_path),
        file_size=os.path.getsize(file_path)
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # 4. Process and Chunk Text
    try:
        chunks = process_document(file_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        
        # Save chunks to database
        for idx, content in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=db_doc.id,
                chunk_index=idx,
                content=content,
                vector_embedding=None  # Embedding is computed later during model training
            )
            db.add(db_chunk)
            
        db.commit()
    except Exception as e:
        # Clean up database entry and file on failure
        db.delete(db_doc)
        db.commit()
        if file_path.exists():
            file_path.unlink()
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse and chunk document: {str(e)}"
        )

    return db_doc


@router.get("", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """
    Retrieves all uploaded documents from the database.
    """
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """
    Deletes a document from the database (along with its chunks via cascade)
    and removes its raw file from storage.
    """
    db_doc = db.query(Document).filter(Document.id == document_id).first()
    if not db_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    # Remove the file from storage
    file_path = Path(db_doc.file_path)
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error removing raw file {file_path}: {e}")

    # Remove database entries (Cascade automatically deletes associated chunks)
    db.delete(db_doc)
    db.commit()
    return
