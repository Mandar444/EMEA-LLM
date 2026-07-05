import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import re
from typing import List, Tuple, Dict

from backend.ml.tokenizers import Tokenizer

class SelfAttention(nn.Module):
    """
    Self-Attention pooling layer.
    Computes importance weights for each token output from the LSTM
    and aggregates them into a single sequence vector.
    """
    def __init__(self, hidden_dim: int):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, lstm_outputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        # mask: [batch_size, seq_len] with 1 for active tokens, 0 for PAD tokens
        
        # Project outputs to attention scores: [batch_size, seq_len, 1]
        scores = self.projection(lstm_outputs)
        
        if mask is not None:
            # Apply large negative value to padded tokens so they get 0 weight in softmax
            scores = scores.squeeze(-1)  # [batch_size, seq_len]
            scores = scores.masked_fill(~mask, -1e9)
            weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
            weights = weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        else:
            weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

        # Context vector: sum(weights * outputs) -> [batch_size, hidden_dim]
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context


class SentenceEncoder(nn.Module):
    """
    Encodes token sequences into 128-dimensional dense vectors.
    Uses frozen pre-trained Word2Vec embeddings, a Bi-LSTM layer,
    and Self-Attention pooling.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, word2vec_weights: torch.Tensor = None):
        super(SentenceEncoder, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if word2vec_weights is not None:
            self.embedding.weight.data.copy_(word2vec_weights)
            # Freeze embeddings to prevent overfitting on small local datasets
            self.embedding.weight.requires_grad = False
            
        # Bi-LSTM Layer
        # hidden_dim = embedding_dim // 2 (since bidirectional=True, output dim = 2 * hidden_dim = embedding_dim)
        lstm_hidden = embedding_dim // 2
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-Attention Layer
        self.attention = SelfAttention(embedding_dim)
        
        # Projection Head to output space
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Create attention mask for non-padding tokens (input_ids != 0)
        mask = (input_ids != 0)
        
        # Embed: [batch_size, seq_len, embedding_dim]
        embeds = self.embedding(input_ids)
        
        # LSTM: [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embeds)
        
        # Attention pooling: [batch_size, embedding_dim]
        pooled_out = self.attention(lstm_out, mask)
        
        # Project: [batch_size, embedding_dim]
        out = self.fc(pooled_out)
        
        # L2 Normalize vectors so cosine similarity equals dot product
        normalized_out = F.normalize(out, p=2, dim=1)
        return normalized_out


class SiameseDataset(Dataset):
    """
    Dataset for training the Siamese network.
    Contains triplets of (Anchor/Query, Candidate Passage, Label),
    where label is 1 for positive match and 0 for negative mismatch.
    """
    def __init__(self, pairs: List[Tuple[List[int], List[int], float]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_ids, passage_ids, label = self.pairs[idx]
        return (
            torch.tensor(query_ids, dtype=torch.long),
            torch.tensor(passage_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function using Cosine Distance.
    Pushes positive pairs close and forces negative pairs apart by a margin.
    """
    def __init__(self, margin: float = 0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Compute cosine similarity between the L2 normalized representations
        cosine_sim = torch.sum(output1 * output2, dim=1)  # Since L2-normalized, dot product is cosine similarity
        
        # Cosine distance: range [0, 2] where 0 is identical, 2 is opposite
        distance = 1.0 - cosine_sim
        
        # Loss formula: y * d^2 + (1-y) * max(0, margin - d)^2
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1.0 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        return torch.mean(loss_pos + loss_neg)


def generate_synthetic_pairs(
    chunks: List[str],
    tokenizer: Tokenizer,
    max_length: int = 120
) -> List[Tuple[List[int], List[int], float]]:
    """
    Generates synthetic query-passage positive and negative pairs for training.
    For each chunk:
      - Positive: construct pseudo-queries by extracting key actions or sentences.
      - Negative: pair the query with a random chunk from the list.
    """
    pairs = []
    num_chunks = len(chunks)
    if num_chunks < 2:
        return pairs
        
    for i, chunk in enumerate(chunks):
        # Extract sentences from the chunk using simple punctuation split
        sentences = [s.strip() for s in re.split(r'[.!?]+', chunk) if len(s.strip()) > 15]
        
        if not sentences:
            continue
            
        # Select up to 2 sentences to act as target answers
        selected_sentences = sentences[:2]
        
        for sent in selected_sentences:
            # Construct a pseudo-query:
            # 1. Clean the sentence
            # 2. Add question templates if it starts with actionable words
            query_candidate = sent.strip()
            
            # Simple templates to make it look like a user query
            if query_candidate.lower().startswith(("to ", "for ", "by ")):
                # e.g., "To restart the server..." -> "How to restart the server?"
                query = re.sub(r"^to\s+", "how to ", query_candidate, flags=re.IGNORECASE)
                if not query.endswith("?"):
                    query += "?"
            else:
                # Add a generic question indicator
                query = "what is " + query_candidate if len(query_candidate) < 40 else query_candidate
                
            query_ids = tokenizer.encode(query, max_length=max_length)
            passage_ids = tokenizer.encode(chunk, max_length=max_length)
            
            # 1. Positive pair (y = 1.0)
            pairs.append((query_ids, passage_ids, 1.0))
            
            # 2. Negative pair (y = 0.0)
            # Find a random chunk that is not the current chunk
            neg_idx = random.choice([idx for idx in range(num_chunks) if idx != i])
            neg_passage_ids = tokenizer.encode(chunks[neg_idx], max_length=max_length)
            pairs.append((query_ids, neg_passage_ids, 0.0))
            
    return pairs


def train_siamese_network(
    pairs: List[Tuple[List[int], List[int], float]],
    vocab_size: int,
    embedding_dim: int,
    word2vec_weights: torch.Tensor,
    epochs: int,
    batch_size: int = 32,
    lr: float = 0.001,
    margin: float = 0.5,
    device: str = "cpu"
) -> SentenceEncoder:
    """
    Trains the shared Siamese Sentence Encoder using Contrastive Loss.
    """
    dataset = SiameseDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    encoder = SentenceEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        word2vec_weights=word2vec_weights
    ).to(device)
    
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    
    encoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for queries, passages, labels in dataloader:
            queries, passages, labels = queries.to(device), passages.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Encode queries and passages through the shared encoder
            q_embeds = encoder(queries)
            p_embeds = encoder(passages)
            
            # Compute contrastive loss
            loss = criterion(q_embeds, p_embeds, labels)
            
            # Optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * queries.size(0)
            
        avg_loss = total_loss / len(dataset)
        print(f"Siamese Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    return encoder
