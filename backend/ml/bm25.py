import math
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple
import numpy as np

class BM25:
    """pi
    NumPy-based BM25 (Best Matching 25) ranking engine.
    Computes exact keyword matching scores over tokenized document chunks.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        
        self.doc_lens: np.ndarray = np.array([])
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def fit(self, corpus_tokens: List[List[str]]) -> None:
        """
        Fits BM25 statistics (frequencies, document lengths, and IDF weights) 
        on a list of tokenized documents.
        """
        self.corpus_size = len(corpus_tokens)
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.doc_lens = np.array([])
            self.doc_freqs = []
            self.idf = {}
            return

        # Calculate document lengths
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens])
        self.avg_doc_len = float(np.mean(self.doc_lens))
        
        # Calculate term frequencies per document
        self.doc_freqs = []
        doc_containing_term: Dict[str, int] = {}
        
        for doc in corpus_tokens:
            frequencies: Dict[str, int] = {}
            unique_terms = set(doc)
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for term in unique_terms:
                doc_containing_term[term] = doc_containing_term.get(term, 0) + 1

        # Calculate Inverse Document Frequency (IDF) using standard BM25 formula
        self.idf = {}
        for term, doc_count in doc_containing_term.items():
            # BM25 IDF formulation with smoothing: log((N - n + 0.5) / (n + 0.5) + 1)
            numerator = self.corpus_size - doc_count + 0.5
            denominator = doc_count + 0.5
            self.idf[term] = float(math.log(max(numerator / denominator, 0.0001) + 1.0))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """
        Computes the BM25 relevance score for each document in the corpus
        against the given query tokens.
        """
        if self.corpus_size == 0:
            return np.zeros(0)

        scores = np.zeros(self.corpus_size)
        doc_len_normalization = (1.0 - self.b) + self.b * (self.doc_lens / (self.avg_doc_len or 1.0))
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            token_idf = self.idf[token]
            
            # Extract term frequency for this token in all documents
            tf = np.array([freqs.get(token, 0) for freqs in self.doc_freqs])
            
            # BM25 score term: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * doc_len_normalization
            scores += token_idf * (numerator / denominator)
            
        return scores

    def save_state(self, file_path: Union[str, Path]) -> None:
        """
        Saves BM25 internal state data to a JSON file.
        """
        state = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens.tolist(),
            "doc_freqs": self.doc_freqs,
            "idf": self.idf
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load_state(self, file_path: Union[str, Path]) -> None:
        """
        Loads BM25 internal state data from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.k1 = state["k1"]
        self.b = state["b"]
        self.corpus_size = state["corpus_size"]
        self.avg_doc_len = state["avg_doc_len"]
        self.doc_lens = np.array(state["doc_lens"])
        self.doc_freqs = state["doc_freqs"]
        self.idf = state["idf"]
