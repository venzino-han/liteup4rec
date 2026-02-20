import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import json
from tqdm import tqdm
import os
from datetime import datetime
import gc
import time

from item_emb_encoder import (
    ItemEmbeddingTransformer,
    load_user_sequences,
    create_dataloaders
)

def infonce_loss(query_emb, positive_emb, negatives_emb, temperature=0.07):
    """
    query: [B, D]
    positive: [B, D]
    negatives: [B, K, D]
    """
    query_emb = F.normalize(query_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negatives_emb = F.normalize(negatives_emb, dim=-1)

    pos_sim = torch.sum(query_emb * positive_emb, dim=1, keepdim=True)
    neg_sim = torch.bmm(negatives_emb, query_emb.unsqueeze(2)).squeeze(2)  # [B, K]
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature

    labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)

def parse_arguments():
    """
    Parse command-line arguments for training the item embedding transformer.
    """
    parser = argparse.ArgumentParser(description="Train item embedding transformer to generate user embeddings.")
    
    # General settings
    parser.add_argument("--run_name", type=str, default="pref")
    parser.add_argument("--data_name", type=str, default="toys")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--infonce_weight", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    
    # Data settings
    parser.add_argument("--item_target_names_str", type=str, default="review_description_target_item")
    parser.add_argument("--model_name", type=str, default="gemma-3-1b-it")
    parser.add_argument("--item_model_name", type=str, default="gemma-3-4b-it")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--token_limit", type=int, default=1024)
    
    args = parser.parse_args()
    args.embedding_model_name_dir = args.embedding_model_name.split("/")[-1]
    
    # Print arguments
    for key, value in vars(args).items():
        print(f"{key:<26}: {value}")
    
    return args


def load_embeddings(args):
    """
    Load item embeddings and user embeddings.
    """
    # Load item embeddings
    item_embedding_file = f"data_emb/{args.data_name}_{args.item_target_names_str}_{args.token_limit}_{args.item_model_name}_{args.embedding_model_name_dir}_emb.pt"
    print(f"Loading item embeddings from: {item_embedding_file}")
    item_embeddings = torch.load(item_embedding_file, map_location='cpu')
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Type of item embeddings: {item_embeddings.dtype}")
    # print(f"Item embeddings: {item_embeddings}")
    
    # Load user embeddings
    train_user_embedding_file = f"data_emb/user_preference_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_pred_emb.pt"
    val_user_embedding_file = f"data_emb/user_preference_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_pred_emb.pt"
    test_user_embedding_file = f"data_emb/user_preference_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_pred_emb.pt"
    
    print(f"Loading train user embeddings from: {train_user_embedding_file}")
    train_user_embeddings = torch.load(train_user_embedding_file, map_location='cpu')
    print(f"Train user embeddings shape: {train_user_embeddings.shape}")
    
    val_user_embeddings = None
    test_user_embeddings = None
    
    if os.path.exists(val_user_embedding_file):
        print(f"Loading validation user embeddings from: {val_user_embedding_file}")
        val_user_embeddings = torch.load(val_user_embedding_file, map_location='cpu')
        print(f"Validation user embeddings shape: {val_user_embeddings.shape}")
    
    if os.path.exists(test_user_embedding_file):
        print(f"Loading test user embeddings from: {test_user_embedding_file}")
        test_user_embeddings = torch.load(test_user_embedding_file, map_location='cpu')
        print(f"Test user embeddings shape: {test_user_embeddings.shape}")
    
    return item_embeddings, train_user_embeddings, val_user_embeddings, test_user_embeddings


def train_epoch(args,model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        item_emb_seqs = batch['item_emb_seqs'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        target_embeddings = batch['target_embeddings'].to(device)
        pos_item_embeddings = batch['pos_item_embeddings'].to(device)
        neg_item_embeddings = batch['neg_item_embeddings'].to(device)
        
        # Forward pass
        predicted_embeddings = model(item_emb_seqs, attention_masks)
        
        # Compute loss
        loss = args.mse_weight * criterion(predicted_embeddings, target_embeddings)
        loss -= args.cosine_weight * torch.cosine_similarity(predicted_embeddings, target_embeddings, dim=-1).mean()
        loss += args.infonce_weight * infonce_loss(predicted_embeddings, pos_item_embeddings, neg_item_embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_epoch(args, model, val_loader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            item_emb_seqs = batch['item_emb_seqs'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            target_embeddings = batch['target_embeddings'].to(device)
            pos_item_embeddings = batch['pos_item_embeddings'].to(device)
            neg_item_embeddings = batch['neg_item_embeddings'].to(device)
            
            # Forward pass
            predicted_embeddings = model(item_emb_seqs, attention_masks)
            
            # Compute loss
            loss = args.mse_weight * criterion(predicted_embeddings, target_embeddings)
            loss -= args.cosine_weight * torch.cosine_similarity(predicted_embeddings, target_embeddings, dim=-1).mean()
            loss += args.infonce_weight * infonce_loss(predicted_embeddings, pos_item_embeddings, neg_item_embeddings)

            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_model(args, model, train_loader, val_loader, device):
    """
    Train the model with early stopping.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(args, model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate_epoch(args, model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            # scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_save_path = f"models/{args.run_name}_{args.data_name}_best.pt"
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args_dict': vars(args)
                }, model_save_path)
                print(f"Saved best model to {model_save_path}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{args.early_stop}")
                
                if patience_counter >= args.early_stop:
                    print("Early stopping triggered!")
                    break
        else:
            # No validation set, save model every few epochs
            if (epoch + 1) % 5 == 0:
                model_save_path = f"models/{args.run_name}_{args.data_name}_epoch_{epoch+1}.pt"
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'args_dict': vars(args)
                }, model_save_path)
    
    return model


def extract_user_embeddings(args, model, dataloader, split_name, device):
    """
    Extract user embeddings using the trained model.
    """
    model.eval()
    
    extracted_embeddings = []
    user_ids = []
    print("Extracting embeddings...")
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            item_emb_seqs = batch['item_emb_seqs'].to(device)
            attention_masks = batch['attention_masks'].to(device)
            batch_user_ids = batch['user_ids']
            
            # Forward pass
            predicted_embeddings = model(item_emb_seqs, attention_masks)
            
            extracted_embeddings.append(predicted_embeddings.cpu())
            user_ids.extend(batch_user_ids)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    # Concatenate all embeddings
    extracted_embeddings = torch.cat(extracted_embeddings, dim=0)
    
    # Save embeddings
    output_file = f"data_emb/{args.run_name}_{args.token_limit}_item_transformer_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_{split_name}_emb.pt"
    torch.save(extracted_embeddings, output_file)
    print(f"Saved {split_name} embeddings to: {output_file}")
    print(f"Embeddings shape: {extracted_embeddings.shape}")

    # mean and std of extracted_embeddings
    print(f"Mean of extracted_embeddings: {extracted_embeddings.mean()}")
    print(f"Std of extracted_embeddings: {extracted_embeddings.std()}")
    print(f"Mean of extracted_embeddings: {extracted_embeddings[10,:].mean()}")
    print(f"Std of extracted_embeddings: {extracted_embeddings[10,:].std()}")

    
    return extracted_embeddings, user_ids


def main():
    """
    Main training and extraction function.
    """
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load embeddings
    item_embeddings, train_user_embeddings, val_user_embeddings, test_user_embeddings = load_embeddings(args)

    # normalize item embeddings
    args.embedding_dim = item_embeddings.shape[1]

        # mean and std of extracted_embeddings
    print(f"Mean of extracted_embeddings: {item_embeddings.mean()}")
    print(f"Std of extracted_embeddings: {item_embeddings.std()}")
    print(f"Mean of extracted_embeddings: {item_embeddings[10,:].mean()}")
    print(f"Std of extracted_embeddings: {item_embeddings[10,:].std()}")
    
    print("Loading user sequences...")
    train_user_sequences, train_pos_item_ids, train_neg_item_sets = load_user_sequences(args.data_name, "train")
    val_user_sequences, val_pos_item_ids, val_neg_item_sets = load_user_sequences(args.data_name, "valid")
    test_user_sequences, test_pos_item_ids, test_neg_item_sets = load_user_sequences(args.data_name, "test")
    print(f"Loaded {len(train_user_sequences)} train user sequences")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args, 
        item_embeddings, 
        train_user_sequences, val_user_sequences, test_user_sequences,
        train_pos_item_ids, val_pos_item_ids, test_pos_item_ids,
        train_neg_item_sets, val_neg_item_sets, test_neg_item_sets,
        train_user_embeddings, val_user_embeddings, test_user_embeddings,
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    if val_loader:
        print(f"Validation loader: {len(val_loader)} batches")
    if test_loader:
        print(f"Test loader: {len(test_loader)} batches")
    
    # Create model
    model = ItemEmbeddingTransformer(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model = train_model(args, model, train_loader, val_loader, device)
    
    # Load best model if validation was used
    if val_loader is not None:
        best_model_path = f"models/{args.run_name}_{args.data_name}_best.pt"
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
            model.to(args.device)
    
    # Extract embeddings for all splits
    print("\nExtracting user embeddings...")
    
    # Train split
    extract_user_embeddings(args, model, train_loader, "train", device)
    
    # Validation split
    if val_loader is not None:
        extract_user_embeddings(args, model, val_loader, "valid", device)
    
    # Test split
    if test_loader is not None:
        extract_user_embeddings(args, model, test_loader, "test", device)
    
    print("Training and extraction completed!")


if __name__ == "__main__":
    main() 