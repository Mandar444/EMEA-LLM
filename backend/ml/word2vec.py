import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import numpy as np

from backend.ml.tokenizers import Tokenizer

class Word2VecDataset(Dataset):
    """
    PyTorch Dataset for Skip-Gram Word2Vec.
    Generates (center_word, context_word) index pairs from tokenized text sequences.
    """
    def __init__(self, tokenized_texts: List[List[int]], window_size: int = 5, pad_idx: int = 0):
        self.pairs: List[Tuple[int, int]] = []
        
        for tokens in tokenized_texts:
            # Filter out padding tokens if they exist
            seq = [t for t in tokens if t != pad_idx]
            seq_len = len(seq)
            
            for i, center_word in enumerate(seq):
                # Context boundaries
                start = max(0, i - window_size)
                end = min(seq_len, i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_word = seq[j]
                        self.pairs.append((center_word, context_word))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class SkipGramModel(nn.Module):
    """
    Skip-Gram Neural Network Model.
    Predicts the surrounding context words given a target center word.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramModel, self).__init__()
        # Target/Center word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Linear layer mapping embeddings back to vocabulary logits for Cross-Entropy loss
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, center_words: torch.Tensor) -> torch.Tensor:
        # Embed center words: [batch_size] -> [batch_size, embedding_dim]
        embeds = self.embeddings(center_words)
        # Project to vocabulary size logits: [batch_size, embedding_dim] -> [batch_size, vocab_size]
        logits = self.output_layer(embeds)
        return logits

    def get_embeddings(self) -> torch.Tensor:
        """
        Returns the trained embedding weight tensor.
        """
        return self.embeddings.weight.data.clone()


def train_word2vec(
    tokenized_texts: List[List[int]],
    vocab_size: int,
    embedding_dim: int,
    window_size: int,
    epochs: int,
    batch_size: int = 128,
    lr: float = 0.01,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Trains the custom Word2Vec Skip-Gram model on tokenized corpora.
    Returns the trained word embedding matrix.
    """
    dataset = Word2VecDataset(tokenized_texts, window_size=window_size)
    
    # If no training pairs exist (corpus too small), return randomly initialized weights
    if len(dataset) == 0:
        model = SkipGramModel(vocab_size, embedding_dim)
        return model.get_embeddings()
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for center, context in dataloader:
            center, context = center.to(device), context.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(center)
            
            # Compute cross entropy
            loss = criterion(logits, context)
            
            # Backward pass & optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * center.size(0)
            
        avg_loss = total_loss / len(dataset)
        print(f"Word2Vec Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    return model.get_embeddings()
