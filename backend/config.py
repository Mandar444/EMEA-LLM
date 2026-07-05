import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
STORAGE_DIR = WORKSPACE_DIR / "storage"

# Create storage directories if they do not exist
RAW_DOCS_DIR = STORAGE_DIR / "raw_documents"
MODEL_CHECKPOINTS_DIR = STORAGE_DIR / "model_checkpoints"

for directory in [RAW_DOCS_DIR, MODEL_CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
# Using SQLite for a lightweight, zero-dependency offline setup
DATABASE_URL = f"sqlite:///{STORAGE_DIR}/assistant.db"

# Text Chunking Settings
CHUNK_SIZE = 600      # Target character length per chunk (~100-150 words)
CHUNK_OVERLAP = 150   # Overlapping characters between consecutive chunks

# Machine Learning Hyperparameters
EMBEDDING_DIM = 128     # Dimensions of both Word2Vec and Siamese embeddings
WORD2VEC_WINDOW = 5     # Word2Vec context window size
WORD2VEC_MIN_COUNT = 1  # Minimum word occurrence to keep in vocabulary
WORD2VEC_EPOCHS = 15    # Number of Word2Vec training epochs

SIAMESE_EPOCHS = 30     # Siamese training epochs
SIAMESE_BATCH_SIZE = 32 # Training batch size
LEARNING_RATE = 0.001   # Optimizer learning rate
MARGIN = 0.5            # Margin for Contrastive Loss function

# Retrieval Engine Settings
RETRIEVAL_ALPHA = 0.6   # Weight for BM25 (0.6) vs Semantic Vector Search (0.4)
CONFIDENCE_THRESHOLD = 0.52  # Matches below this score trigger the safety fallback message

# Fallback Message
FALLBACK_MESSAGE = (
    "I am sorry, but I cannot find an answer to your question in the provided documentation."
)
