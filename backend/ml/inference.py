import re
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from backend.config import (
    MODEL_CHECKPOINTS_DIR,
    EMBEDDING_DIM,
    RETRIEVAL_ALPHA,
    CONFIDENCE_THRESHOLD,
    FALLBACK_MESSAGE
)
from backend.ml.tokenizers import Tokenizer
from backend.ml.bm25 import BM25
from backend.ml.siamese_lstm import SentenceEncoder
from backend.ml.answer_generator import AnswerGenerator
from backend.ml.query_understanding import QueryUnderstandingLayer

class InferenceCoordinator:
    """
    Coordinator class that manages inference pipelines.
    Loads trained vocabularies, BM25 parameters, and Siamese Bi-LSTM weights.
    Executes hybrid (lexical + semantic) query search and checks confidence thresholds.
    """
    def __init__(self, embedding_dim: Optional[int] = None):
        self.vocab_path = MODEL_CHECKPOINTS_DIR / "vocab.json"
        self.bm25_path = MODEL_CHECKPOINTS_DIR / "bm25.json"
        self.encoder_path = MODEL_CHECKPOINTS_DIR / "siamese_encoder.pt"
        
        self.tokenizer: Optional[Tokenizer] = None
        self.bm25: Optional[BM25] = None
        self.encoder: Optional[SentenceEncoder] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self.query_understanding: Optional[QueryUnderstandingLayer] = None
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.model_loaded = False

    def load_models(self) -> bool:
        """
        Loads the tokenizer, BM25 state, and Siamese encoder weights from checkpoints.
        Returns True if successful, False if checkpoints are missing.
        """
        if not (self.vocab_path.exists() and self.bm25_path.exists() and self.encoder_path.exists()):
            self.model_loaded = False
            return False
            
        try:
            # 1. Load Tokenizer & Vocab
            self.tokenizer = Tokenizer()
            self.tokenizer.load_vocab(self.vocab_path)
            
            # 2. Load BM25 state
            self.bm25 = BM25()
            self.bm25.load_state(self.bm25_path)
            
            # 3. Load PyTorch Siamese Encoder
            # Instantiate SentenceEncoder (Word2Vec weights are not needed for pure inference,
            # we load the complete model weights directly)
            state_dict = torch.load(self.encoder_path, map_location=torch.device("cpu"))
            if "embedding.weight" in state_dict:
                self.embedding_dim = state_dict["embedding.weight"].shape[1]
                
            self.encoder = SentenceEncoder(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=self.embedding_dim,
                word2vec_weights=None
            )
            self.encoder.load_state_dict(state_dict)
            self.encoder.eval()
            
            # 4. Load Query Understanding Layer & Answer Generator
            self.query_understanding = QueryUnderstandingLayer(self.tokenizer.word2idx)
            self.answer_generator = AnswerGenerator(self.tokenizer, self.encoder, self.bm25)
            self.model_loaded = True
            print("Inference models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading inference models: {e}")
            self.model_loaded = False
            return False

    def compute_query_vector(self, query_text: str) -> np.ndarray:
        """
        Tokenizes and embeds query text using the Siamese encoder.
        Returns a L2 normalized 1D NumPy array representing the query.
        """
        if not self.model_loaded or self.tokenizer is None or self.encoder is None:
            raise RuntimeError("Models are not loaded. Call load_models() first.")
            
        query_ids = self.tokenizer.encode(query_text, max_length=120, padding=True)
        query_tensor = torch.tensor([query_ids], dtype=torch.long)
        
        with torch.no_grad():
            query_vector = self.encoder(query_tensor)
            
        return query_vector.squeeze(0).cpu().numpy()

    @staticmethod
    def normalize_retrieval_query(query_text: str) -> Tuple[str, List[str]]:
        """
        Minimal retrieval normalization only:
        lowercase, remove punctuation, normalize whitespace, and return word tokens.
        """
        normalized = query_text.lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        tokens = normalized.split() if normalized else []
        return normalized, tokens

    def search(
        self,
        query_text: str,
        database_chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> Tuple[str, float, Optional[Dict[str, Any]], str]:
        """
        Performs hybrid retrieval (BM25 + Dense Semantic) on candidate database chunks.
        Applies a confidence threshold check:
          - If the top score exceeds the threshold, returns the matched chunk.
          - If not, returns a standard fallback message to prevent hallucinations.
        
        Returns:
            Tuple: (response_text, score, matched_chunk_info_or_None, retrieval_source)
        """
        if not self.model_loaded:
            return "Assistant is not trained yet. Please upload documents and initiate training.", 0.0, None, "system"
            
        if not database_chunks:
            return FALLBACK_MESSAGE, 0.0, None, "system"
            
        # 1. Minimal retrieval normalization.
        # Do not apply spell correction, synonym expansion, fuzzy replacement,
        # abbreviation expansion, or any token substitution before retrieval.
        retrieval_query, query_tokens = self.normalize_retrieval_query(query_text)

        # 2. BM25 Lexical Scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Max-normalize BM25 scores safely to [0, 1] range
        max_bm25 = np.max(bm25_scores)
        if max_bm25 > 0:
            norm_bm25_scores = bm25_scores / max_bm25
        else:
            norm_bm25_scores = np.zeros_like(bm25_scores)
            
        # 3. Semantic Similarity Scores
        query_vector = self.compute_query_vector(retrieval_query)
        
        semantic_scores = []
        for chunk in database_chunks:
            # Parse stored JSON embedding array
            embedding_str = chunk.get("vector_embedding")
            if embedding_str:
                chunk_vector = np.array(json.loads(embedding_str), dtype=np.float32)
                # Compute Cosine Similarity (dot product since they are L2-normalized)
                sim = np.dot(query_vector, chunk_vector)
                semantic_scores.append(float(sim))
            else:
                semantic_scores.append(0.0)
                
        semantic_scores = np.array(semantic_scores)
        
        # 4. Combine scores using weighted average
        # Score = alpha * BM25 + (1 - alpha) * Semantic
        search_alpha = alpha if alpha is not None else RETRIEVAL_ALPHA
        combined_scores = search_alpha * norm_bm25_scores + (1.0 - search_alpha) * semantic_scores
        
        # 5. Filter chunks that pass the threshold and gather top K (up to 10 for better coverage)
        target_threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
        
        # Check if the highest score passes the threshold
        best_idx = int(np.argmax(combined_scores)) if len(combined_scores) > 0 else 0
        best_score = float(combined_scores[best_idx]) if len(combined_scores) > 0 else 0.0
        
        # Adaptive check for partial answerability
        is_partial = False
        if best_score < target_threshold and best_score >= max(0.3, target_threshold - 0.08):
            is_partial = True
            
        if best_score < target_threshold and not is_partial:
            return FALLBACK_MESSAGE, best_score, None, "fallback"
            
        # Select passing chunks (or if partial, we take the top chunk)
        if is_partial:
            passing_indices = [best_idx]
        else:
            passing_indices = [i for i, score in enumerate(combined_scores) if score >= target_threshold]
            passing_indices.sort(key=lambda idx: combined_scores[idx], reverse=True)
            
        top_k_indices = passing_indices[:10]
        selected_chunks = []
        for idx in top_k_indices:
            chunk_dict = dict(database_chunks[idx])
            chunk_dict["score"] = float(combined_scores[idx])
            selected_chunks.append(chunk_dict)
        
        # Determine primary source details
        best_chunk = database_chunks[best_idx]
        primary_source_info = {
            "chunk_id": best_chunk.get("id"),
            "document_id": best_chunk.get("document_id"),
            "filename": best_chunk.get("filename"),
            "chunk_index": best_chunk.get("chunk_index")
        }
        
        # Determine if match was primary lexical or semantic for auditing
        source_type = "hybrid"
        if norm_bm25_scores[best_idx] > 0.8 and semantic_scores[best_idx] < 0.4:
            source_type = "lexical"
        elif semantic_scores[best_idx] > 0.8 and norm_bm25_scores[best_idx] < 0.4:
            source_type = "semantic"
            
        # Generate the concise, structured answer
        if self.answer_generator is not None:
            generated_text, contributing_sources = self.answer_generator.generate_answer(
                query_text=query_text,
                query_vector=query_vector,
                retrieved_chunks=selected_chunks,
                threshold=target_threshold,
                fallback_message=FALLBACK_MESSAGE
            )
            primary_source_info["contributing_sources"] = contributing_sources
            if is_partial:
                # Strip generated title and prepend partial match callout
                generated_text_no_title = re.sub(r'^# .*\n\n', '', generated_text)
                title = f"# Partial Match Reference\n\n"
                warning = "> [!NOTE]\n> The documentation does not explicitly answer your question, but contains related terms. Here is the closest match:\n\n"
                generated_text = title + warning + generated_text_no_title
                
            return generated_text, best_score, primary_source_info, source_type
        else:
            ans = best_chunk.get("content")
            if is_partial:
                ans = f"# Partial Match Reference\n\n> [!NOTE]\n> Close match text:\n\n" + ans
            return ans, best_score, primary_source_info, source_type

# Singleton instance for application-wide reuse
inference_coordinator = InferenceCoordinator()
