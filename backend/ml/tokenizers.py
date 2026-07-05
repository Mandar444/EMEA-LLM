import re
import json
from pathlib import Path
from typing import List, Dict, Union

class Tokenizer:
    """
    A custom word-level tokenizer and vocabulary manager.
    Handles text normalization, tokenization, vocabulary construction,
    sequence encoding (with padding/truncation), and file state persistence.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self):
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN
        }
        self.vocab_size = 2

    @staticmethod
    def clean_and_split(text: str) -> List[str]:
        """
        Normalizes text and splits it into lowercase words and individual symbols.
        For example, "hello, world!" becomes ["hello", ",", "world", "!"]
        """
        if not text:
            return []
        # Convert to lowercase
        text = text.lower()
        # Find all word sequences OR individual non-word, non-whitespace symbol characters
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def build_vocab(self, texts: List[str], min_count: int = 1) -> None:
        """
        Builds the vocabulary from a list of text strings based on frequency counts.
        """
        # Count word frequencies
        word_counts: Dict[str, int] = {}
        for text in texts:
            tokens = self.clean_and_split(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Reset vocab to base tokens
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID
        }
        self.idx2word = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN
        }
        self.vocab_size = 2
        
        # Filter by min_count and assign indices
        for word, count in word_counts.items():
            if count >= min_count and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text: str, max_length: int = None, padding: bool = True) -> List[int]:
        """
        Converts text into a list of vocabulary token IDs.
        Optionally pads or truncates the list to match max_length.
        """
        tokens = self.clean_and_split(text)
        token_ids = [self.word2idx.get(token, self.UNK_ID) for token in tokens]
        
        if max_length is not None:
            if len(token_ids) > max_length:
                # Truncate
                token_ids = token_ids[:max_length]
            elif len(token_ids) < max_length and padding:
                # Pad
                padding_length = max_length - len(token_ids)
                token_ids.extend([self.PAD_ID] * padding_length)
                
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Converts a list of token IDs back into a space-separated string of words.
        Filters out PAD tokens.
        """
        words = []
        for token_id in token_ids:
            if token_id == self.PAD_ID:
                continue
            words.append(self.idx2word.get(token_id, self.UNK_TOKEN))
        return " ".join(words)

    def save_vocab(self, file_path: Union[str, Path]) -> None:
        """
        Saves the vocabulary mappings to a JSON file.
        """
        data = {
            "word2idx": self.word2idx,
            "vocab_size": self.vocab_size
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_vocab(self, file_path: Union[str, Path]) -> None:
        """
        Loads the vocabulary mappings from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.word2idx = data["word2idx"]
        self.vocab_size = data["vocab_size"]
        
        # Reconstruct idx2word
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
