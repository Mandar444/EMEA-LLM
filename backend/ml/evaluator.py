import os
import re
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal
from backend.db.models import DocumentChunk, Document
from backend.ml.inference import InferenceCoordinator
from backend.config import MODEL_CHECKPOINTS_DIR, FALLBACK_MESSAGE

class ResponseEvaluator:
    """
    Automated response quality evaluation framework.
    Measures Retrieval Accuracy, Sentence Precision/Recall, Hallucination Rate, and Citation Correctness.
    """
    def __init__(self, coordinator: InferenceCoordinator):
        self.coordinator = coordinator
        self.db = SessionLocal()
        
        # 1. Hardcoded high-quality evaluation cases based on the user guide
        self.predefined_cases = [
            {
                "query": "How do I log in to the MySAF-T system?",
                "expected_chunk_index": 2,
                "keywords": ["email", "password", "log in", "field"]
            },
            {
                "query": "What is the purpose of the user guide?",
                "expected_chunk_index": 1,
                "keywords": ["instructions", "interface", "capabilities", "scenarios"]
            },
            {
                "query": "Where is the side navigation menu located?",
                "expected_chunk_index": 4,
                "keywords": ["left", "modules", "quick switching"]
            },
            {
                "query": "How do you navigate between sections?",
                "expected_chunk_index": 5,
                "keywords": ["side menu", "quick action buttons", "navigation"]
            },
            {
                "query": "What functional blocks are contained in the home page?",
                "expected_chunk_index": 8,
                "keywords": ["navigation panel", "logo", "period indicator", "profile"]
            },
            {
                "query": "How can a user select the global reporting period?",
                "expected_chunk_index": 7,
                "keywords": ["logging in", "window", "reporting period"]
            },
            {
                "query": "How do I access the Settings panel?",
                "expected_chunk_index": 13,
                "keywords": ["Settings", "side navigation menu", "settings page"]
            },
            {
                "query": "What is the company settings section designed for?",
                "expected_chunk_index": 17,
                "keywords": ["parameters", "processing", "report generation"]
            },
            {
                "query": "Who has access to the user management section?",
                "expected_chunk_index": 19,
                "keywords": ["administrator", "creating new users", "roles"]
            },
            {
                "query": "How do you save changes in the field settings form?",
                "expected_chunk_index": 15,
                "keywords": ["Save", "Reset", "Go to table"]
            },
            {
                "query": "What does the user profile section allow you to do?",
                "expected_chunk_index": 12,
                "keywords": ["view information", "own account", "role"]
            },
            {
                "query": "How do I go to the user profile settings?",
                "expected_chunk_index": 11,
                "keywords": ["profile icon", "navigation panel"]
            },
            {
                "query": "What is the main function of the side navigation menu?",
                "expected_chunk_index": 4,
                "keywords": ["switching", "modules", "navigation menu"]
            },
            {
                "query": "How do you configure the company parameters in steps?",
                "expected_chunk_index": 18,
                "keywords": ["Fill in", "Next", "Back"]
            },
            {
                "query": "What does the Settings panel do?",
                "expected_chunk_index": 13,
                "keywords": ["system settings", "administrator"]
            },
            {
                "query": "Where do users view their role in the system?",
                "expected_chunk_index": 12,
                "keywords": ["profile", "role", "account"]
            },
            {
                "query": "What is the quick actions block designed for?",
                "expected_chunk_index": 9,
                "keywords": ["data import", "tables", "generating", "frequently"]
            },
            {
                "query": "Who can delete user accounts in the system?",
                "expected_chunk_index": 19,
                "keywords": ["administrator", "users"]
            },
            {
                "query": "What is the function of the Save button in field settings?",
                "expected_chunk_index": 15,
                "keywords": ["saves", "changes"]
            },
            {
                "query": "What does the top navigation panel contain?",
                "expected_chunk_index": 8,
                "keywords": ["logo", "period indicator", "profile icon"]
            }
        ]

    def close(self):
        self.db.close()

    def generate_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """
        Dynamically compiles a suite of 100+ evaluation cases.
        Loads chunks from the database, extracts specific sentences using intent-based patterns,
        and builds evaluation queries paired with ground-truth chunk contexts.
        """
        # Fetch document chunks from DB
        chunks = self.db.query(DocumentChunk).order_by(DocumentChunk.chunk_index.asc()).all()
        if not chunks:
            return []

        dataset = []
        
        # Add predefined cases, resolving expected chunk IDs from DB matching keywords dynamically
        for case in self.predefined_cases:
            matching_chunk = None
            # Find chunk containing the case keywords to support dynamic shifting chunks
            for c in chunks:
                c_content_lower = c.content.lower()
                if all(kw.lower() in c_content_lower for kw in case["keywords"]):
                    matching_chunk = c
                    break
            if not matching_chunk:
                matching_chunk = next((c for c in chunks if c.chunk_index == case["expected_chunk_index"]), None)
                
            if matching_chunk:
                dataset.append({
                    "query": case["query"],
                    "expected_chunk_id": matching_chunk.id,
                    "expected_chunk_index": matching_chunk.chunk_index,
                    "ground_truth_sentence": matching_chunk.content,
                    "keywords": case["keywords"]
                })

        # Procedural rule-based generation to complete the 100+ case dataset
        sentence_end = re.compile(
            r'(?<!\b[A-Za-z]\.)'
            r'(?<!\b\d\.)'
            r'(?<!\b(?:eq|vs|eg|ie|dr|mr|ms|v4|v3|v2|v1)\.)'
            r'(?<!\b(?:vol)\.)'
            r'(?<=\.|\?|\!)\s+'
        )

        for chunk in chunks:
            sentences = sentence_end.split(chunk.content)
            for s in sentences:
                s = s.strip()
                if not s or len(s) < 25:
                    continue
                
                query_item = None
                
                # Rule 1: "To [X], perform/do [Y]" -> "How to [X]?"
                to_match = re.search(r'^[tT]o\s+([a-zA-Z\s]{4,40}),\s+([a-zA-Z\s]+)', s)
                if to_match:
                    action = to_match.group(1).strip()
                    if not any(stop in action.lower() for stop in ["click", "select", "enter"]):
                        query_item = {
                            "query": f"How do I {action}?",
                            "ground_truth_sentence": s,
                            "expected_chunk_id": chunk.id,
                            "expected_chunk_index": chunk.chunk_index,
                            "keywords": [w.lower() for w in action.split() if len(w) > 3]
                        }

                # Rule 2: "[Subject] is designed for [Y]" -> "What is [Subject] designed for?"
                if not query_item:
                    des_match = re.search(r'^([A-Z][a-zA-Z\s]{3,30})\s+is\s+designed\s+for\s+([a-zA-Z\s]+)', s)
                    if des_match:
                        subject = des_match.group(1).strip()
                        purpose = des_match.group(2).strip()
                        query_item = {
                            "query": f"What is the {subject} designed for?",
                            "ground_truth_sentence": s,
                            "expected_chunk_id": chunk.id,
                            "expected_chunk_index": chunk.chunk_index,
                            "keywords": [w.lower() for w in subject.split() if len(w) > 3]
                        }

                # Rule 3: "Only [Role] has/have access..." -> "Who has access to [Topic]?"
                if not query_item:
                    access_match = re.search(r'^[oO]nly\s+([a-zA-Z\s]{3,30})\s+has\s+access\s+to\s+([a-zA-Z\s]+)', s)
                    if not access_match:
                        access_match = re.search(r'^[oO]nly\s+([a-zA-Z\s]{3,30})\s+have\s+access\s+to\s+([a-zA-Z\s]+)', s)
                    if access_match:
                        role = access_match.group(1).strip()
                        section = access_match.group(2).strip()
                        query_item = {
                            "query": f"Who has access to {section}?",
                            "ground_truth_sentence": s,
                            "expected_chunk_id": chunk.id,
                            "expected_chunk_index": chunk.chunk_index,
                            "keywords": [w.lower() for w in role.split() if len(w) > 3]
                        }

                # Rule 4: "[Subject] is a [definition]" -> "What is [Subject]?"
                if not query_item:
                    def_match = re.search(r'^([A-Z][a-zA-Z\s]{3,30})\s+is\s+a\s+([a-zA-Z\s]+)', s)
                    if def_match:
                        subject = def_match.group(1).strip()
                        query_item = {
                            "query": f"What is the {subject}?",
                            "ground_truth_sentence": s,
                            "expected_chunk_id": chunk.id,
                            "expected_chunk_index": chunk.chunk_index,
                            "keywords": [w.lower() for w in subject.split() if len(w) > 3]
                        }

                if query_item:
                    # Prevent duplicate queries in evaluation suite
                    if not any(q["query"] == query_item["query"] for q in dataset):
                        dataset.append(query_item)

        # If we need more items to reach 100, add generic keyword sentences
        if len(dataset) < 100:
            for chunk in chunks:
                sentences = sentence_end.split(chunk.content)
                for s in sentences:
                    s = s.strip()
                    if len(s) > 40 and not any(q["ground_truth_sentence"] == s for q in dataset):
                        # Extract first 4 words as a query trigger
                        words = [w for w in re.sub(r'[^\w\s]', '', s).split() if len(w) > 3]
                        if len(words) >= 3:
                            q_text = f"Tell me about {' '.join(words[:3])}."
                            dataset.append({
                                "query": q_text,
                                "ground_truth_sentence": s,
                                "expected_chunk_id": chunk.id,
                                "expected_chunk_index": chunk.chunk_index,
                                "keywords": [w.lower() for w in words[:3]]
                            })
                    if len(dataset) >= 105:
                        break
                if len(dataset) >= 105:
                    break

        return dataset[:102]  # Cap at around 100 items

    def run_eval(self) -> Dict[str, Any]:
        """
        Runs the complete automated evaluation suite and returns metrics.
        """
        # 1. Fetch DB chunks in coordinator format
        chunks_query = self.db.query(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.vector_embedding,
            Document.filename
        ).join(Document, Document.id == DocumentChunk.document_id).all()

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

        if not database_chunks:
            return {
                "success": False,
                "error": "No indexed documents found in database. Please index manuals before evaluating."
            }

        # Load models
        if not self.coordinator.model_loaded:
            self.coordinator.load_models()

        if not self.coordinator.model_loaded:
            return {
                "success": False,
                "error": "Model checkpoints are missing or could not be loaded."
            }

        # 2. Compile evaluation dataset
        eval_set = self.generate_evaluation_dataset()
        if not eval_set:
            return {
                "success": False,
                "error": "Could not generate evaluation dataset from indexed files."
            }

        total_queries = len(eval_set)
        retrieval_hits = 0
        precision_scores = []
        recall_scores = []
        hallucination_rates = []
        citation_correctness_scores = []
        queries_evaluated = []

        # Token cleaning helper
        def get_clean_words(text: str) -> List[str]:
            # Remove markdown, tags, punctuation, and citation tags like [1]
            text_cleaned = re.sub(r'\[\d+\]', '', text)
            text_cleaned = re.sub(r'[^\w\s]', ' ', text_cleaned)
            return [w.lower() for w in text_cleaned.split() if w.strip()]

        for item in eval_set:
            query = item["query"]
            expected_chunk_id = item["expected_chunk_id"]
            expected_chunk_index = item["expected_chunk_index"]
            ground_truth = item["ground_truth_sentence"]
            
            # Execute search
            generated_ans, score, source_info, source_type = self.coordinator.search(
                query_text=query,
                database_chunks=database_chunks
            )
            
            # 1. Retrieval Accuracy
            # Correct if it matches the chunk index or if the retrieved chunk contains the ground truth
            retrieved_correct = False
            retrieved_chunk_index = source_info["chunk_index"] if source_info else None
            retrieved_chunk_id = source_info["chunk_id"] if source_info else None
            
            if retrieved_chunk_index == expected_chunk_index:
                retrieved_correct = True
            elif source_info and "contributing_sources" in source_info and source_info["contributing_sources"]:
                contrib_idxs = [c["chunk_index"] for c in source_info["contributing_sources"]]
                if expected_chunk_index in contrib_idxs:
                    retrieved_correct = True
            
            if not retrieved_correct:
                # Fallback check: find the retrieved chunk content
                retrieved_chunk_obj = next((c for c in database_chunks if c["id"] == retrieved_chunk_id), None)
                if retrieved_chunk_obj and ground_truth in retrieved_chunk_obj["content"]:
                    retrieved_correct = True
                    
            if retrieved_correct:
                retrieval_hits += 1

            # 2. Sentence Precision & Recall
            # Strip title and bibliography from answer to focus on generated text content
            ans_body = generated_ans
            # Remove title
            ans_body = re.sub(r'^# .*\n\n', '', ans_body)
            # Remove bibliography
            bib_split = ans_body.split("\n\nSources:\n")
            ans_body_text = bib_split[0]
            
            gen_words = get_clean_words(ans_body_text)
            gt_words = get_clean_words(ground_truth)
            
            gen_word_set = set(gen_words)
            gt_word_set = set(gt_words)
            
            intersection = gen_word_set.intersection(gt_word_set)
            
            precision = len(intersection) / len(gen_word_set) if gen_word_set else 0.0
            recall = len(intersection) / len(gt_word_set) if gt_word_set else 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            # 3. Hallucination Rate
            # Check what percentage of generated words are NOT found in ANY of the retrieved chunks
            # Find the actual retrieved chunks matched from source_info
            contrib = source_info.get("contributing_sources") if source_info else None
            retrieved_chunk_contents = ""
            if contrib:
                retrieved_chunk_contents = " ".join([
                    next((c["content"] for c in database_chunks if c["id"] == src["chunk_id"]), "")
                    for src in contrib
                ])
            else:
                retrieved_chunk_id = source_info["chunk_id"] if source_info else None
                retrieved_chunk_contents = next((c["content"] for c in database_chunks if c["id"] == retrieved_chunk_id), "")
                
            retrieved_words_set = set(get_clean_words(retrieved_chunk_contents))
            
            hallucinated_words = [w for w in gen_words if w not in retrieved_words_set]
            # Ignore standard formatting keywords
            hallucinated_words = [w for w in hallucinated_words if w not in [
                "yes", "no", "partially", "context", "dependent", "match", "closest", "explicitly", "answer", "warning", 
                "conflict", "detected", "according", "category", "documentation", "potentials", "statements", "manual", "regarding"
            ]]
            
            hall_rate = len(hallucinated_words) / len(gen_words) if gen_words else 0.0
            hallucination_rates.append(hall_rate)
            
            # 4. Citation Correctness
            # Verify citation markers exist and match bibliography
            citation_correct = 1.0
            citation_markers = re.findall(r'\[(\d+)\]', ans_body_text)
            
            bib_matches = []
            if len(bib_split) > 1:
                bib_lines = bib_split[1].split("\n")
                for line in bib_lines:
                    match = re.match(r'^\[(\d+)\]\s+(.+?)\s+\(Chunk\s+#(\d+)\)', line.strip())
                    if match:
                        bib_matches.append({
                            "index": int(match.group(1)),
                            "filename": match.group(2),
                            "chunk_index": int(match.group(3))
                        })
            
            if citation_markers:
                marker_ints = [int(m) for m in citation_markers]
                bib_ints = [b["index"] for b in bib_matches]
                
                # Check if all markers in text are in bibliography
                for m in marker_ints:
                    if m not in bib_ints:
                        citation_correct = 0.0
                        break
                        
                # Check if bibliography matches the actual retrieved chunks
                if contrib:
                    retrieved_filenames = [src["filename"] for src in contrib]
                    retrieved_chunk_idxs = [src["chunk_index"] for src in contrib]
                else:
                    retrieved_filenames = [source_info["filename"]] if source_info else []
                    retrieved_chunk_idxs = [source_info["chunk_index"]] if source_info else []
                
                for b in bib_matches:
                    if b["filename"] not in retrieved_filenames or b["chunk_index"] not in retrieved_chunk_idxs:
                        citation_correct = 0.0
                        break
            else:
                if generated_ans == FALLBACK_MESSAGE or "The provided documentation does not contain" in generated_ans:
                    citation_correct = 1.0
                else:
                    citation_correct = 0.0
                    
            citation_correctness_scores.append(citation_correct)
            
            # Save query summary
            queries_evaluated.append({
                "query": query,
                "retrieved_chunk_index": retrieved_chunk_index,
                "expected_chunk_index": expected_chunk_index,
                "retrieval_accuracy": 1.0 if retrieved_correct else 0.0,
                "precision": precision,
                "recall": recall,
                "hallucination_rate": hall_rate,
                "citation_correctness": citation_correct
            })

        # Calculate final aggregates
        retrieval_accuracy = retrieval_hits / total_queries
        avg_precision = float(np.mean(precision_scores))
        avg_recall = float(np.mean(recall_scores))
        avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0
        avg_hallucination = float(np.mean(hallucination_rates))
        avg_citation_correctness = float(np.mean(citation_correctness_scores))
        
        result_payload = {
            "success": True,
            "total_queries": total_queries,
            "retrieval_accuracy": retrieval_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "hallucination_rate": avg_hallucination,
            "citation_correctness": avg_citation_correctness,
            "queries": queries_evaluated
        }
        
        # Save results to local storage cache
        cache_path = Path("c:/Users/smand/OneDrive/Desktop/EMEAit llm/storage/evaluation_results.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2)
        except Exception as e:
            print(f"Error caching evaluation results: {e}")
            
        return result_payload

def run_evaluation():
    print("Initializing evaluation run...")
    coord = InferenceCoordinator()
    evaluator = ResponseEvaluator(coord)
    res = evaluator.run_eval()
    print("Evaluation results:")
    print(f"Total Queries: {res.get('total_queries')}")
    print(f"Retrieval Accuracy: {res.get('retrieval_accuracy', 0.0) * 100:.2f}%")
    print(f"Precision: {res.get('precision', 0.0) * 100:.2f}%")
    print(f"Recall: {res.get('recall', 0.0) * 100:.2f}%")
    print(f"F1 Score: {res.get('f1_score', 0.0) * 100:.2f}%")
    print(f"Hallucination Rate: {res.get('hallucination_rate', 0.0) * 100:.2f}%")
    print(f"Citation Correctness: {res.get('citation_correctness', 0.0) * 100:.2f}%")
    evaluator.close()

if __name__ == "__main__":
    run_evaluation()
