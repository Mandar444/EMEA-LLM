from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
import json
import torch
import traceback
import zipfile
import io

from backend.config import (
    EMBEDDING_DIM,
    WORD2VEC_WINDOW,
    WORD2VEC_EPOCHS,
    SIAMESE_EPOCHS,
    LEARNING_RATE,
    MARGIN,
    MODEL_CHECKPOINTS_DIR
)
from backend.core.database import SessionLocal, get_db
from backend.db.models import DocumentChunk
from backend.db.schemas import TrainingStatusResponse
from backend.ml.tokenizers import Tokenizer
from backend.ml.word2vec import train_word2vec
from backend.ml.siamese_lstm import generate_synthetic_pairs, train_siamese_network
from backend.ml.bm25 import BM25
from backend.ml.inference import inference_coordinator

router = APIRouter(prefix="/training", tags=["training"])

# Global in-memory training status tracker
training_state = {
    "status": "idle",       # "idle", "training", "completed", "failed"
    "progress": 0,          # 0 to 100
    "error": None,
    "last_trained": None
}

def train_worker():
    """
    Background worker that runs the complete pipeline:
    1. Rebuilds vocabulary from database chunks.
    2. Trains custom Word2Vec embedding vectors.
    3. Generates synthetic training pairs.
    4. Trains Siamese Bi-LSTM model with Attention.
    5. Fits BM25 indexes.
    6. Precomputes and caches embedding vectors inside SQL database.
    7. Reloads coordinator instance to make models live.
    """
    global training_state
    db: Session = SessionLocal()
    
    try:
        # Update state
        training_state["status"] = "training"
        training_state["progress"] = 5
        training_state["error"] = None
        
        # Load all chunks
        chunks = db.query(DocumentChunk).all()
        if len(chunks) < 2:
            raise ValueError("You must upload documents and generate at least 2 chunks before training.")
            
        chunk_texts = [c.content for c in chunks]
        
        # --- Step 1: Build Vocabulary ---
        training_state["progress"] = 15
        tokenizer = Tokenizer()
        tokenizer.build_vocab(chunk_texts, min_count=1)
        
        vocab_path = MODEL_CHECKPOINTS_DIR / "vocab.json"
        tokenizer.save_vocab(vocab_path)
        
        # --- Step 2: Train Word2Vec Skip-Gram ---
        training_state["progress"] = 25
        tokenized_texts = [tokenizer.encode(t, padding=False) for t in chunk_texts]
        
        w2v_weights = train_word2vec(
            tokenized_texts=tokenized_texts,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            window_size=WORD2VEC_WINDOW,
            epochs=WORD2VEC_EPOCHS,
            device="cpu"
        )
        
        # --- Step 3: Generate Synthetic Pairs & Train Siamese Network ---
        training_state["progress"] = 45
        pairs = generate_synthetic_pairs(chunk_texts, tokenizer, max_length=120)
        
        if len(pairs) < 2:
            raise ValueError(
                "Unable to generate synthetic contrastive QA pairs. "
                "Ensure uploaded documents contain longer, complete sentences."
            )
            
        training_state["progress"] = 55
        encoder = train_siamese_network(
            pairs=pairs,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            word2vec_weights=w2v_weights,
            epochs=SIAMESE_EPOCHS,
            lr=LEARNING_RATE,
            margin=MARGIN,
            device="cpu"
        )
        
        encoder_path = MODEL_CHECKPOINTS_DIR / "siamese_encoder.pt"
        torch.save(encoder.state_dict(), encoder_path)
        
        # --- Step 4: Fit and Save BM25 Indices ---
        training_state["progress"] = 75
        corpus_tokens = [tokenizer.clean_and_split(txt) for txt in chunk_texts]
        bm25 = BM25()
        bm25.fit(corpus_tokens)
        
        bm25_path = MODEL_CHECKPOINTS_DIR / "bm25.json"
        bm25.save_state(bm25_path)
        
        # --- Step 5: Precompute Chunk Embeddings ---
        training_state["progress"] = 85
        encoder.eval()
        for chunk in chunks:
            chunk_ids = tokenizer.encode(chunk.content, max_length=120, padding=True)
            chunk_tensor = torch.tensor([chunk_ids], dtype=torch.long)
            
            with torch.no_grad():
                vector_tensor = encoder(chunk_tensor)
                
            # Serialize L2-normalized float list to JSON string
            vector_list = vector_tensor.squeeze(0).cpu().numpy().tolist()
            chunk.vector_embedding = json.dumps(vector_list)
            
        db.commit()
        
        # --- Step 6: Reload Inference Coordinator ---
        training_state["progress"] = 95
        inference_coordinator.load_models()
        
        training_state["status"] = "completed"
        training_state["progress"] = 100
        training_state["last_trained"] = datetime.now().isoformat()
        print("Training worker completed successfully.")
        
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        training_state["status"] = "failed"
        training_state["error"] = str(e)
    finally:
        db.close()


@router.post("/start", response_model=TrainingStatusResponse)
def start_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Triggers the custom machine learning model training loop in an asynchronous background thread.
    """
    global training_state
    
    # Check if training is already in progress
    if training_state["status"] == "training":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training is already in progress."
        )
        
    # Check if there are documents to train on
    chunk_count = db.query(DocumentChunk).count()
    if chunk_count < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents found to train on. Please upload files first."
        )

    # Fire background task
    background_tasks.add_task(train_worker)
    
    training_state["status"] = "training"
    training_state["progress"] = 0
    training_state["error"] = None
    
    return training_state


@router.get("/status", response_model=TrainingStatusResponse)
def get_training_status():
    """
    Retrieves the current training progress, state, and errors of the background training pipeline.
    """
    global training_state
    return training_state


@router.post("/rebuild-embeddings")
def rebuild_embeddings(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Recalculates the dense semantic vector embeddings for all document chunks.
    Does not retrain Word2Vec or Siamese model weights.
    """
    if not inference_coordinator.model_loaded or inference_coordinator.encoder is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Active model checkpoints are not loaded. Train models first."
        )
        
    def worker():
        db_session = SessionLocal()
        try:
            tokenizer = inference_coordinator.tokenizer
            encoder = inference_coordinator.encoder
            encoder.eval()
            
            chunks = db_session.query(DocumentChunk).all()
            for chunk in chunks:
                chunk_ids = tokenizer.encode(chunk.content, max_length=120, padding=True)
                chunk_tensor = torch.tensor([chunk_ids], dtype=torch.long)
                with torch.no_grad():
                    vector_tensor = encoder(chunk_tensor)
                vector_list = vector_tensor.squeeze(0).cpu().numpy().tolist()
                chunk.vector_embedding = json.dumps(vector_list)
            db_session.commit()
            print("Successfully completed vector embedding rebuild.")
        except Exception as e:
            db_session.rollback()
            print(f"Error rebuilding embeddings: {e}")
        finally:
            db_session.close()
            
    background_tasks.add_task(worker)
    return {"message": "Rebuilding embeddings started in background worker."}


@router.post("/rebuild-bm25")
def rebuild_bm25(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Fits and saves the BM25 model indexes for the current chunks.
    Does not require model retraining.
    """
    if not inference_coordinator.model_loaded or inference_coordinator.tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Active tokenizer is not loaded. Train models first."
        )
        
    def worker():
        db_session = SessionLocal()
        try:
            tokenizer = inference_coordinator.tokenizer
            chunks = db_session.query(DocumentChunk).all()
            chunk_texts = [c.content for c in chunks]
            
            corpus_tokens = [tokenizer.clean_and_split(txt) for txt in chunk_texts]
            bm25 = BM25()
            bm25.fit(corpus_tokens)
            
            bm25_path = MODEL_CHECKPOINTS_DIR / "bm25.json"
            bm25.save_state(bm25_path)
            inference_coordinator.load_models()
            print("Successfully completed BM25 index rebuild.")
        except Exception as e:
            print(f"Error rebuilding BM25: {e}")
        finally:
            db_session.close()
            
    background_tasks.add_task(worker)
    return {"message": "Rebuilding BM25 started in background worker."}


@router.get("/export")
def export_model():
    """
    Compresses model checkpoint files (vocab.json, siamese_encoder.pt, bm25.json)
    into a single ZIP file for export.
    """
    vocab_path = MODEL_CHECKPOINTS_DIR / "vocab.json"
    encoder_path = MODEL_CHECKPOINTS_DIR / "siamese_encoder.pt"
    bm25_path = MODEL_CHECKPOINTS_DIR / "bm25.json"
    
    if not (vocab_path.exists() and encoder_path.exists() and bm25_path.exists()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model checkpoints are missing. Please complete training first."
        )
        
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(vocab_path, "vocab.json")
        zip_file.write(encoder_path, "siamese_encoder.pt")
        zip_file.write(bm25_path, "bm25.json")
        
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=offline_model.zip"}
    )


@router.post("/import")
def import_model(file: UploadFile = File(...)):
    """
    Extracts uploaded zip containing checkpoints directly into model directory,
    re-initializes active models.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid format. Please upload a ZIP model archive."
        )
        
    try:
        contents = file.file.read()
        zip_buffer = io.BytesIO(contents)
        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            namelist = zip_file.namelist()
            required = ["vocab.json", "siamese_encoder.pt", "bm25.json"]
            if not all(r in namelist for r in required):
                raise ValueError("ZIP archive is missing checkpoints: vocab.json, siamese_encoder.pt, or bm25.json")
                
            zip_file.extract("vocab.json", MODEL_CHECKPOINTS_DIR)
            zip_file.extract("siamese_encoder.pt", MODEL_CHECKPOINTS_DIR)
            zip_file.extract("bm25.json", MODEL_CHECKPOINTS_DIR)
            
        success = inference_coordinator.load_models()
        if not success:
            raise RuntimeError("Coordinator failed to load the extracted checkpoints.")
            
        return {"status": "success", "message": "Checkpoints imported and loaded successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Checkpoints import failed: {str(e)}"
        )
