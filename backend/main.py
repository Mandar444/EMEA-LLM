import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.core.database import engine, Base
from backend.ml.inference import inference_coordinator
from backend.api import documents, training, chat, system

# Initialize SQLite database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Enterprise Offline AI Assistant",
    description="Offline domain-specific QA system using custom word embeddings and semantic retrieval.",
    version="1.0.0"
)

# Configure CORS Middleware
# Allows the React frontend to communicate with Uvicorn API server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to the frontend origin (e.g., http://localhost:5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Event
@app.on_event("startup")
def startup_event():
    """
    Tries to load pre-trained checkpoints into the InferenceCoordinator
    so the assistant is immediately functional if a model exists.
    """
    print("FastAPI Application Booting...")
    loaded = inference_coordinator.load_models()
    if loaded:
        print("Success: Loaded models into memory on startup.")
    else:
        print("Notice: No model checkpoints found. Please upload documents and train.")

# Mount API Routers
app.include_router(documents.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(system.router, prefix="/api")

# Root Health Route
@app.get("/")
def health_check():
    """
    Basic endpoint to confirm server health and model status.
    """
    return {
        "status": "healthy",
        "model_loaded": inference_coordinator.model_loaded,
        "message": "Enterprise AI Assistant Backend is active."
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
