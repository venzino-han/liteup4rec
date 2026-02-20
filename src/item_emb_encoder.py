import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import random

class ItemEmbeddingTransformer(nn.Module):
    """
    Transformer encoder that takes item embedding sequence and generates user embedding.
    The sequence is reversed so that the most recent item comes first.
    """
    
    def __init__(self, 
                 embedding_dim: int = 1024,
                 num_heads: int = 1,
                 num_layers: int = 1,
                 max_seq_length: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        # self.projection = nn.Linear(embedding_dim, embedding_dim//2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, item_embeddings: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        
        Args:
            item_embeddings: (batch_size, seq_len, embedding_dim)
            attention_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
            
        Returns:
            user_embedding: (batch_size, embedding_dim) - using the first (most recent) position
        """
        batch_size, seq_len, _ = item_embeddings.shape
        
        # Create positional embeddings
        positions = torch.arange(seq_len, device=item_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embedding(positions)
        
        # Add positional embeddings to item embeddings
        x = item_embeddings + pos_embeddings
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=item_embeddings.device)
        
        # Convert to transformer format (True for valid tokens, False for padding)
        transformer_mask = attention_mask.bool()
        
        # Pass through transformer encoder
        # Note: transformer expects src_key_padding_mask where True means ignore
        transformer_output = self.transformer(x, src_key_padding_mask=~transformer_mask)
        
        # Use the first position (most recent item) as the user embedding
        user_embedding = transformer_output[:, 0, :]
        
        # Apply output projection
        user_embedding = self.output_projection(user_embedding)
        
        return user_embedding


class ItemEmbeddingDataset(Dataset):
    """
    Dataset for training the item embedding transformer.
    Loads user sequences and corresponding target user embeddings.
    """
    
    def __init__(self, 
                 user_sequences: Dict[int, List[int]],
                 item_embeddings: torch.Tensor,
                 pos_item_ids: Dict[int, int],
                 neg_item_sets: Dict[int, List[int]],
                 target_user_embeddings: torch.Tensor,
                 max_seq_length: int = 8,
    ):
        """
        Args:
            user_sequences: Dict mapping user_id to list of item_ids
            item_embeddings: (num_items, embedding_dim) tensor of item embeddings
            target_user_embeddings: (num_users, embedding_dim) tensor of target user embeddings
            max_seq_length: Maximum sequence length
            history_limit: Number of recent items to consider
        """
        self.user_sequences = user_sequences
        self.item_embeddings = item_embeddings
        self.pos_item_ids = pos_item_ids
        self.neg_item_sets = neg_item_sets
        self.target_user_embeddings = target_user_embeddings
        self.max_seq_length = max_seq_length
        
        # Create user_id to index mapping
        self.user_ids = list(user_sequences.keys())
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_sequence = self.user_sequences[user_id]
        pos_item_id = self.pos_item_ids[user_id]
        neg_item_set = self.neg_item_sets[user_id]

        # get pos item embedding
        pos_item_embedding = self.item_embeddings[pos_item_id]
        neg_item_embeddings = self.item_embeddings[neg_item_set]

        # Take the most recent items up to history_limit
        recent_items = item_sequence[-self.max_seq_length:]
        
        # Reverse the sequence so most recent comes first
        recent_items = recent_items[::-1]
        
        # Pad or truncate to max_seq_length
        if len(recent_items) > self.max_seq_length:
            recent_items = recent_items[:self.max_seq_length]
        else:
            # Pad with zeros (will be masked)
            recent_items = recent_items + [0] * (self.max_seq_length - len(recent_items))
        
        # Convert to tensor
        item_sequence_tensor = torch.tensor(recent_items, dtype=torch.long)
        
        # Get corresponding item embeddings
        item_emb_seq = self.item_embeddings[item_sequence_tensor]
        
        # Create attention mask (1 for valid items, 0 for padding)
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
        valid_length = min(len(item_sequence[-self.max_seq_length:]), self.max_seq_length)
        attention_mask[valid_length:] = 0
        
        # Get target user embedding
        target_embedding = self.target_user_embeddings[user_id]
        
        return {
            'item_emb_seq': item_emb_seq,
            'attention_mask': attention_mask,
            'target_embedding': target_embedding,
            'user_id': user_id,
            'pos_item_embedding': pos_item_embedding,
            'neg_item_embeddings': neg_item_embeddings,
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    item_emb_seqs = torch.stack([item['item_emb_seq'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    target_embeddings = torch.stack([item['target_embedding'] for item in batch])
    user_ids = [item['user_id'] for item in batch]
    pos_item_embeddings = torch.stack([item['pos_item_embedding'] for item in batch])
    neg_item_embeddings = torch.stack([item['neg_item_embeddings'] for item in batch])
    return {
        'item_emb_seqs': item_emb_seqs,
        'attention_masks': attention_masks,
        'target_embeddings': target_embeddings,
        'user_ids': user_ids,
        'pos_item_embeddings': pos_item_embeddings,
        'neg_item_embeddings': neg_item_embeddings,
    }


def load_user_sequences(data_name: str, split: str) -> Dict[int, List[int]]:
    """
    Load user sequences from the dataset.
    """
    if split == "train":
        target_index = -3
    elif split == "valid":
        target_index = -2
    elif split == "test":
        target_index = -1

    data_file = f"data/{data_name}/sequential_data.txt"
    user_sequences = {}
    pos_item_ids = {}
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            parts = [int(p) for p in parts]
            user_id = parts[0]
            user_sequences[user_id] = parts[1:target_index]
            pos_item_ids[user_id] = parts[target_index]

    # negative sampling for training
    data_file = f"data/{data_name}/negative_samples.txt"
    neg_item_sets = {}
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            parts = [int(p) for p in parts]
            user_id = parts[0]
            parts = parts[1:]

            # randomly sample 10 negative items
            if len(parts) > 99:
                parts = random.sample(parts, 99)
            neg_item_sets[user_id] = parts

    return user_sequences, pos_item_ids, neg_item_sets


def create_dataloaders(
        args, 
        item_embeddings, 
        train_user_sequences, val_user_sequences, test_user_sequences,
        train_pos_item_ids, val_pos_item_ids, test_pos_item_ids,
        train_neg_item_sets, val_neg_item_sets, test_neg_item_sets,
        train_user_embeddings, val_user_embeddings, test_user_embeddings,                   
    ):
    """
    Create train, validation, and test dataloaders.
    """
    # Split user sequences based on available user embeddings
    user_ids = set(range(1, len(train_user_embeddings) + 1))

    train_sequences = {uid: train_user_sequences[uid] for uid in train_user_sequences.keys() 
                      if uid in user_ids}
    val_sequences = {uid: val_user_sequences[uid] for uid in val_user_sequences.keys() 
                      if uid in user_ids}
    test_sequences = {uid: test_user_sequences[uid] for uid in test_user_sequences.keys() 
                      if uid in user_ids}
    
    # Create datasets
    train_dataset = ItemEmbeddingDataset(
        train_sequences,
        item_embeddings,
        train_pos_item_ids,
        train_neg_item_sets,
        train_user_embeddings,
        max_seq_length=args.max_seq_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    

    val_dataset = ItemEmbeddingDataset(
        val_sequences,
        item_embeddings,
        val_pos_item_ids,
        val_neg_item_sets,
        val_user_embeddings,
        max_seq_length=args.max_seq_length,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    test_dataset = ItemEmbeddingDataset(
        test_sequences,
        item_embeddings,
        test_pos_item_ids,
        test_neg_item_sets,
        test_user_embeddings,
        max_seq_length=args.max_seq_length,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader 