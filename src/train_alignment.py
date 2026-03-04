from typing import Dict

import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
import random
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim

from utils.common import (
    calculate_hit_rate,
    calculate_ndcg,
    )

from copy import deepcopy

def parse_arguments():
    """
    Parse command-line arguments for inference and training settings.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a transformer encoder model.")

    # General settings
    parser.add_argument("--run_name", type=str, default="basic_fusion")
    parser.add_argument("--data_name", type=str, default="toys")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--check_interval", type=int, default=2000)

    parser.add_argument("--token_limit", type=int, default=1024, help="Max token limit for the model.")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot learning.")
    
    # Model settings
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--use_llm_emb", action="store_true", help="Use LLM embeddings.")

    # Fusion settings
    parser.add_argument("--cf_model", type=str, default="SASRec", help="Type of CF model to use.")
    parser.add_argument("--use_fusion_mlp", action="store_true", help="Use fusion MLP.")
    parser.add_argument("--fusion_method", type=str, default="concat", help="Fusion method to use.") # add, concat, dot, mul
    parser.add_argument("--item_wise_weight", action="store_true", help="Use item-wise weight.")
    parser.add_argument("--llm_rec_weight", type=float, default=1.0)
    parser.add_argument("--wo_item_weight", action="store_true", help="Use item-wise weight.")
    parser.add_argument("--wo_user_weight", action="store_true", help="Use user-wise weight.")
    parser.add_argument("--wo_recsys", action="store_true", help="without recsys")
    parser.add_argument("--use_hit", action="store_true", help="Use hit item embeddings.")
    parser.add_argument("--use_weight_gate", action="store_true", help="Use weight gate.")
    parser.add_argument("--late_fusion", action="store_true", help="Use late fusion: train without CRM, eval with CRM concat.")

    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--alpha", type=str, default="0.0")
    parser.add_argument("--beta", type=str, default="0.0")
    parser.add_argument("--gamma", type=str, default="1.0")
    parser.add_argument("--cf_penalty_ratio", type=float, default=1.0)
    parser.add_argument("--days", type=int, default=14, help="Number of days to consider for history.")

    parser.add_argument("--item_target_names_str", type=str, default="review_description")
    parser.add_argument("--use_item_meta", action="store_true", help="Use item meta information.")
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    
    parser.add_argument("--history_limit", type=int, default=8, help="Limit the number of history items to use.")
    parser.add_argument("--num_neg_samples", type=int, default=30, help="Number of negative samples to use.")
    parser.add_argument("--use_origin_emb", action="store_true", help="Use origin embeddings.")

    parser.add_argument("--model_name", type=str, default="gemma-3-1b-it", help="Name of the model to use.")
    parser.add_argument("--item_model_name", type=str, default="gemma-3-4b-it", help="Name of the model to use.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--target_split", type=str, default="test", help="Target split to evaluate on.")

    args = parser.parse_args()
    args.beta = float(args.beta)
    args.alpha = float(args.alpha)
    args.gamma = float(args.gamma)
    args.embedding_model_name_dir = args.embedding_model_name.split("/")[-1]

    # if args.add_item_meta:
    #     args.embedding_dim = args.embedding_dim * 2

    for key, value in vars(args).items():
        print(f"{key:<26}: {value}")

    return args

def load_cf_embeddings(args):
    train_user_embs_path = f'crm_emb/{args.cf_model}_{args.data_name}_train_user_embeddings.pt'
    valid_user_embs_path = f'crm_emb/{args.cf_model}_{args.data_name}_val_user_embeddings.pt'
    test_user_embs_path = f'crm_emb/{args.cf_model}_{args.data_name}_test_user_embeddings.pt'
    item_embs_path = f'crm_emb/{args.cf_model}_{args.data_name}_item_embeddings.pt'
    train_user_embs = torch.load(train_user_embs_path)
    valid_user_embs = torch.load(valid_user_embs_path)
    test_user_embs = torch.load(test_user_embs_path)
    item_embs = torch.load(item_embs_path)

    # to device
    train_user_embs = train_user_embs.detach().to(args.device)
    valid_user_embs = valid_user_embs.detach().to(args.device)
    test_user_embs = test_user_embs.detach().to(args.device)
    item_embs = item_embs.detach().to(args.device)

    return train_user_embs, valid_user_embs, test_user_embs, item_embs


# DTYPE = torch.float16
DTYPE = torch.float32

class WeightConstant(nn.Module):
    def __init__(self, constant=0.5):
        super().__init__()
        self.constant = constant

    def forward(self, emb_batch):
        return self.constant

class FusionModel(nn.Module):
    def __init__(
        self, 
        args, 
        llm_emb_dim=1024, cf_emb_dim=128, output_dim=512, 
        max_user_id=35000,
        max_item_id=20000,
        llm_item_embs=None, cf_item_embs=None, item_hit_label=None,
        dropout=0.5, 
        ):

        super().__init__()
        self.args = args
        self.llm_item_embs = llm_item_embs.to(torch.float32).to(args.device)
        self.cf_item_embs = cf_item_embs.to(torch.float32).to(args.device)
        self.item_hit_label = item_hit_label.to(args.device)
        self.dim_ratio = output_dim / cf_emb_dim

        # test penalty ratio for cf 
        self.cf_penalty_ratio = args.cf_penalty_ratio

        # llm projection mlp
        self.llm_mlp = nn.Sequential(
            nn.Linear(llm_emb_dim, output_dim),
        )
        if self.args.zero_shot:
            self.llm_mlp = nn.Identity()

        self.cf_mlp = nn.Sequential(
            nn.Identity(),
        )

        print("max_user_id:", max_user_id)
        print("max_item_id:", max_item_id)

        self.mse_loss = nn.MSELoss()

        if not self.args.zero_shot:
            self.__init_weights()
        else:
            self.__zero_shot_setup()


    def __init_weights(self):
        try:
            nn.init.xavier_uniform_(self.llm_mlp[0].weight)
            # nn.init.xavier_uniform_(self.cf_mlp[0].weight)
        except:
            pass

    def __zero_shot_setup(self):
        self.llm_mlp = nn.Identity()
        self.cf_mlp = nn.Identity()
        self.weight_embedding = nn.Identity()

    def infonce_loss(self, query_emb, positive_emb, negatives_emb, temperature=0.07):
        """
        query: [B, D]
        positive: [B, D]
        negatives: [B, K, D]
        """
        # query_emb = F.normalize(query_emb, dim=-1)
        # positive_emb = F.normalize(positive_emb, dim=-1)
        # negatives_emb = F.normalize(negatives_emb, dim=-1)

        pos_sim = torch.sum(query_emb * positive_emb, dim=1, keepdim=True)
        neg_sim = torch.bmm(negatives_emb, query_emb.unsqueeze(2)).squeeze(2)  # [B, K]
        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature

        labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
        return F.cross_entropy(logits, labels)
        
    def reg_loss(self, query_emb, positive_emb, negatives_emb):
        """
        regularization loss
        average l2 norm 
        """
        return torch.norm(query_emb, p=2)/query_emb.shape[0] + torch.norm(positive_emb, p=2)/positive_emb.shape[0] + torch.norm(negatives_emb, p=2)/(negatives_emb.shape[0]*self.args.num_neg_samples)

    def fusion_embeddings(self, llm_embs, cf_embs, is_item=False, item_hit_label=None, is_training=True):
        llm_embs = self.llm_mlp(llm_embs)
        cf_embs = self.cf_mlp(cf_embs)

        # Late fusion: only use LLM embeddings during training
        if self.args.late_fusion and is_training:
            return llm_embs, llm_embs, cf_embs

        if self.args.wo_recsys:
            return llm_embs, llm_embs, cf_embs
        if self.args.fusion_method == "add":
            return llm_embs + cf_embs, llm_embs, cf_embs
        elif self.args.fusion_method == "concat":
            return torch.cat([llm_embs, cf_embs], dim=-1), llm_embs, cf_embs

    def forward(self, 
            user_ids,
            llm_user_embs, cf_user_embs, 
            pos_item_ids, neg_item_ids, 
            user_cf_llm_hit_label,
            is_training=True,
        ):

        llm_user_embs = llm_user_embs.to(torch.float32)
        cf_user_embs = cf_user_embs.to(torch.float32)
        self.user_cf_llm_hit_label = user_cf_llm_hit_label.unsqueeze(1).to(self.args.device)
        self.loss = 0.0
        
        # Get positive and negative item embeddings
        pos_llm_item_embs = self.llm_item_embs[pos_item_ids]  # [B, D]
        pos_cf_item_embs = self.cf_item_embs[pos_item_ids]    # [B, D]
        neg_llm_item_embs = self.llm_item_embs[neg_item_ids]  # [B, K, D]
        neg_cf_item_embs = self.cf_item_embs[neg_item_ids]    # [B, K, D]

        user_embs, llm_embs, cf_embs = self.fusion_embeddings(llm_user_embs, cf_user_embs, is_training=is_training)
        pos_item_embs, pos_llm_embs, pos_cf_embs = self.fusion_embeddings(pos_llm_item_embs, pos_cf_item_embs, is_item=True, item_hit_label=self.item_hit_label[pos_item_ids], is_training=is_training)
        neg_item_embs, neg_llm_embs, neg_cf_embs = self.fusion_embeddings(neg_llm_item_embs, neg_cf_item_embs, is_item=True, item_hit_label=self.item_hit_label[neg_item_ids], is_training=is_training)

        # llm rec loss 
        llm_rec_loss = self.infonce_loss(llm_embs, pos_llm_embs, neg_llm_embs)
        reg_loss = self.reg_loss(user_embs, pos_item_embs, neg_item_embs)
        self.loss += self.args.alpha * reg_loss
        self.loss += self.args.llm_rec_weight * llm_rec_loss        

        # Compute the InfoNCE loss
        self.loss += self.args.alpha * self.infonce_loss(user_embs, pos_item_embs, neg_item_embs)
        # loss += self.args.cf_rec_weight * cf_rec_loss
        
        
        if self.args.use_origin_emb:
            user_embs = torch.cat([llm_user_embs, user_embs], dim=-1)
        return user_embs, self.loss

    
    def get_item_emb(self, llm_item_embs, cf_item_embs, is_training=False):
        """
        llm_item_embs: [N, D]
        cf_item_embs: [N, D]
        """
        llm_item_embs = llm_item_embs.to(torch.float32)
        cf_item_embs = cf_item_embs.to(torch.float32)
        item_emb, _, cf_item_embs = self.fusion_embeddings(llm_item_embs, cf_item_embs, is_item=True, item_hit_label=self.item_hit_label, is_training=is_training)
        if self.args.use_origin_emb:
            item_emb = torch.cat([llm_item_embs, item_emb], dim=-1)
        return item_emb

def get_dataloader(
        llm_user_embs, cf_user_embs, 
        llm_item_embs, cf_item_embs, 
        pos_item_ids, neg_item_sets, 
        user_cf_llm_hit_label,
        batch_size=128, shuffle=True
        ):
    """
    llm_user_embs: [N, D]
    cf_user_embs: [N, D]
    pos_item_ids: [N,]
    neg_item_ids: [N, K,]
    llm_item_embs: [M, D]
    cf_item_embs: [M, D]
    """
    print("llm_user_embs.shape:", llm_user_embs.shape)
    print("cf_user_embs.shape:", cf_user_embs.shape)
    print("pos_item_ids length:", len(pos_item_ids))
    print("neg_item_sets length:", len(neg_item_sets))

    # remove the first padding from user_embs
    llm_user_embs = llm_user_embs[1:]
    cf_user_embs = cf_user_embs[1:]

    print("llm_user_embs.shape:", llm_user_embs.shape)
    print("cf_user_embs.shape:", cf_user_embs.shape)
    print("pos_item_ids length:", len(pos_item_ids))
    print("neg_item_sets length:", len(neg_item_sets))
    assert llm_user_embs.shape[0] == cf_user_embs.shape[0] == len(pos_item_ids) == len(neg_item_sets), "Embedding length do not match"

    # Convert to tensors
    pos_item_ids = torch.tensor(pos_item_ids, dtype=torch.long)
    neg_item_sets = torch.tensor(neg_item_sets, dtype=torch.long)

    # hit label
    hit_label = torch.ones(llm_user_embs.shape[0]) * 0.5
    # for user_id in user_cf_llm_hit_label:
    #     hit_label[user_id-1] = user_cf_llm_hit_label[user_id]

    user_ids = torch.arange(1, llm_user_embs.shape[0]+1, dtype=torch.long)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_sets, hit_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_rec(
    args, model, train_loader, valid_loader, epochs=10, device="cuda:0"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    model = model.to(device)
    best_model = model

    best_val_loss = float('inf')
    prev_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = datetime.now()
        for batch_idx, (user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label) in enumerate(train_loader):
            user_ids = user_ids.to(device)
            llm_user_embs = llm_user_embs.to(device)
            cf_user_embs = cf_user_embs.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)

            optimizer.zero_grad()
            _, loss = model(user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label, is_training=True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        end_time = datetime.now()
        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_idx, (user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label) in enumerate(valid_loader):
                user_ids = user_ids.to(device)
                llm_user_embs = llm_user_embs.to(device)
                cf_user_embs = cf_user_embs.to(device)
                pos_item_ids = pos_item_ids.to(device)
                neg_item_ids = neg_item_ids.to(device)

                _, val_loss = model(user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label, is_training=False)
                total_val_loss += val_loss.item()
            val_loss = total_val_loss / len(valid_loader)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch time: {end_time - start_time}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model).to("cpu")
        if val_loss > prev_val_loss:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                print("Early stopping triggered.")
                break
        prev_val_loss = val_loss

    return best_model

def test_fusion_model(
    model, test_loader, device="cuda:0",
):  
    model = model.to(device)
    model.eval()
    total_user_embs = []
    with torch.no_grad():
        total_loss = 0
        for batch_idx, (user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label) in tqdm(enumerate(test_loader), total=len(test_loader)):
            user_ids = user_ids.to(device)
            llm_user_embs = llm_user_embs.to(device)
            cf_user_embs = cf_user_embs.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)

            user_embs, loss = model(user_ids, llm_user_embs, cf_user_embs, pos_item_ids, neg_item_ids, hit_label, is_training=False)
            total_loss += loss.item()
            total_user_embs.append(user_embs.cpu())
        total_user_embs = torch.cat(total_user_embs, dim=0)
        print("Total Test user embeddings shape:", total_user_embs.shape)
        test_loss = total_loss / len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")

    return test_loss, total_user_embs, None  # Return None for item_embs since we'll use weighted scoring


def compute_item_popularity(train_user_seq_data, train_user_target):
    """
    Compute item interaction counts from training data.

    Returns:
        item_popularity (dict): item_id -> interaction count
        warm_item_set (set): top-20% popular item IDs (warm)
        total_train_interactions (int): sum of all interaction counts
    """
    item_popularity = {}
    for user_id, seq in train_user_seq_data.items():
        for item_id in seq:
            item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
        target = train_user_target[user_id]
        item_popularity[target] = item_popularity.get(target, 0) + 1

    total_train_interactions = sum(item_popularity.values())

    sorted_items = sorted(item_popularity.keys(), key=lambda x: item_popularity[x], reverse=True)
    warm_cutoff = max(1, int(len(sorted_items) * 0.2))
    warm_item_set = set(sorted_items[:warm_cutoff])

    print(f"[Popularity] Total items: {len(item_popularity)}, "
          f"Warm (top 20%): {len(warm_item_set)}, "
          f"Total interactions: {total_train_interactions}")
    return item_popularity, warm_item_set, total_train_interactions


    # test the model
def evaluate_model(
    model, data_loader, user_seq_data, user_target, device, split_name, args,
    item_popularity=None, warm_item_set=None, total_train_interactions=0,
):
    """
    Evaluate model performance and save results to CSV file (batch processing version).

    Args:
        model: Trained model
        data_loader: DataLoader for the split
        user_seq_data: User sequence data
        user_target: User target data
        device: Device to use
        split_name: Name of the split (train/valid/test)
        args: Arguments
        item_popularity: dict[item_id -> interaction count] from train set
        warm_item_set: set of warm item IDs (top-20% by popularity)
        total_train_interactions: total number of train interactions (for novelty)
    """
    # Get predictions
    loss, pred_user_embs, _ = test_fusion_model(
        model, data_loader, device=str(device)
    )

    print(f"shape of user_pred_embs ({split_name}):", pred_user_embs.shape)

    hit_results = {
        "ndcg@5": [],
        "ndcg@10": [],
        "hit@5": [],
        "hit@10": [],
        "hit@100": [],
        "hit@500": [],
        # novelty of recommendation list (mean log-inverse-popularity of top-k)
        "novelty@5": [],
        "novelty@10": [],
        # novelty-weighted hit/ndcg (target item novelty * hit/ndcg)
        "novelty_hit@5": [],
        "novelty_hit@10": [],
        "novelty_ndcg@5": [],
        "novelty_ndcg@10": [],
    }
    warm_results = {"hit@5": [], "hit@10": [], "ndcg@5": [], "ndcg@10": []}
    cold_results = {"hit@5": [], "hit@10": [], "ndcg@5": [], "ndcg@10": []}

    result_dict_list = []
    num_users = pred_user_embs.shape[0]
    num_items = model.llm_item_embs.shape[0]
    pred_user_embs = pred_user_embs.to(device)  # [N, D]

    pred_item_embs = model.get_item_emb(model.llm_item_embs, model.cf_item_embs, is_training=False)
    print(f"shape of pred_item_embs ({split_name}):", pred_item_embs.shape)

    # Batch compute similarities: [N, M]
    print("Computing similarities for all users...")
    similarities = pred_user_embs @ pred_item_embs.T  # [N, M]

    # Create mask for history items
    print("Creating history mask...")
    mask = torch.zeros((num_users, num_items), dtype=torch.bool, device=device)

    for user_id, history_iids in tqdm(user_seq_data.items(), total=len(user_seq_data)):
        if split_name == "train":
            history_to_mask = history_iids[:-2]
        elif split_name == "valid":
            history_to_mask = history_iids[:-1]
        else:  # test
            history_to_mask = history_iids

        history_to_mask = history_to_mask + [0]
        history_to_mask = list(set(history_to_mask) - set([0]))
        mask[user_id-1, history_to_mask] = True

    # Apply mask: set masked items to -inf
    similarities[mask] = float('-inf')

    # Get top-k items for all users at once
    print("Getting top-k items...")
    top_scores, top_indices = torch.topk(similarities, k=500, dim=1)  # [N, 500]
    top_scores = top_scores.cpu()
    top_indices = top_indices.cpu()

    # Novelty helpers
    use_novelty = (item_popularity is not None) and (total_train_interactions > 1)
    log_total = np.log2(total_train_interactions) if use_novelty else 1.0

    def get_novelty(item_id):
        """log2(N / count) — higher value means less popular (more novel)."""
        count = max(item_popularity.get(item_id, 1), 1)
        return np.log2(total_train_interactions / count)

    # Process each user
    print("Processing results...")
    for user_id, history_iids in user_seq_data.items():
        target_item_id = user_target[user_id]
        user_idx = user_id - 1

        user_top_indices = top_indices[user_idx].tolist()
        user_top_scores = top_scores[user_idx].tolist()

        if target_item_id not in user_top_indices:
            rank = 501
        else:
            rank = user_top_indices.index(target_item_id)

        hit_list = [1 if item == target_item_id else 0 for item in user_top_indices]

        # --- standard metrics ---
        hit5  = calculate_hit_rate(hit_list, 5)
        hit10 = calculate_hit_rate(hit_list, 10)
        hit100 = calculate_hit_rate(hit_list, 100)
        hit500 = calculate_hit_rate(hit_list, 500)
        ndcg5  = calculate_ndcg(hit_list, 5)
        ndcg10 = calculate_ndcg(hit_list, 10)

        hit_results["hit@5"].append(hit5)
        hit_results["hit@10"].append(hit10)
        hit_results["hit@100"].append(hit100)
        hit_results["hit@500"].append(hit500)
        hit_results["ndcg@5"].append(ndcg5)
        hit_results["ndcg@10"].append(ndcg10)

        # --- warm / cold ---
        if warm_item_set is not None:
            bucket = warm_results if target_item_id in warm_item_set else cold_results
            bucket["hit@5"].append(hit5)
            bucket["hit@10"].append(hit10)
            bucket["ndcg@5"].append(ndcg5)
            bucket["ndcg@10"].append(ndcg10)

        # --- novelty ---
        if use_novelty:
            # average novelty of the top-k recommendation list
            for k, hit_results_k in [(5, "novelty@5"), (10, "novelty@10")]:
                avg_nov = float(np.mean([get_novelty(i) for i in user_top_indices[:k]]))
                hit_results[hit_results_k].append(avg_nov)

            # novelty-weighted hit / ndcg: weight = normalized novelty of target item
            target_nov  = get_novelty(target_item_id)
            norm_nov    = target_nov / log_total if log_total > 0 else 0.0

            hit_results["novelty_hit@5"].append(norm_nov * hit5)
            hit_results["novelty_hit@10"].append(norm_nov * hit10)
            hit_results["novelty_ndcg@5"].append(norm_nov * ndcg5)
            hit_results["novelty_ndcg@10"].append(norm_nov * ndcg10)

        # Store per-user result row
        top_item_ids   = " ".join([str(iid)          for iid   in user_top_indices])
        top_scores_str = " ".join([str(round(s, 2))  for s     in user_top_scores])

        if split_name == "train":
            display_history = history_iids[:-2]
        elif split_name == "valid":
            display_history = history_iids[:-1]
        else:
            display_history = history_iids

        result_dict_list.append({
            "user_id": user_id,
            "history_item_ids": " ".join([str(iid) for iid in display_history]),
            "target_item_id": target_item_id,
            "top_item_ids": top_item_ids,
            "top_item_scores": top_scores_str,
            "rank": rank,
        })

        # Print metrics periodically
        if user_id % args.check_interval == 0:
            print(f"Processed {user_id} / {len(user_target)} ({split_name})")
            partial = {k: float(round(np.mean(v), 4)) for k, v in hit_results.items() if v}
            for k in ["hit@5", "hit@10", "ndcg@5", "ndcg@10", "hit@100", "hit@500"]:
                print(f"{k:>10}: {partial.get(k, 0.0):.4f}")
            if use_novelty:
                for k in ["novelty@5", "novelty@10", "novelty_hit@5", "novelty_hit@10",
                          "novelty_ndcg@5", "novelty_ndcg@10"]:
                    print(f"{k:>20}: {partial.get(k, 0.0):.4f}")

    # ---- aggregate ----
    avg_results = {k: float(round(np.mean(v), 4)) for k, v in hit_results.items() if v}
    print(f"Average Metrics ({split_name}):", avg_results)

    if warm_item_set is not None:
        n_warm = len(warm_results["hit@5"])
        n_cold = len(cold_results["hit@5"])
        avg_warm = {f"warm_{k}": float(round(np.mean(v), 4)) if v else 0.0
                    for k, v in warm_results.items()}
        avg_cold = {f"cold_{k}": float(round(np.mean(v), 4)) if v else 0.0
                    for k, v in cold_results.items()}
        print(f"Warm items (n={n_warm}):", avg_warm)
        print(f"Cold items (n={n_cold}):", avg_cold)
        avg_results.update(avg_warm)
        avg_results.update(avg_cold)

    # Save results to CSV file
    results_df = pd.DataFrame(result_dict_list)
    csv_filename = (f"results/embedding_training_{args.data_name}_{args.run_name}"
                    f"_{args.model_name}_{split_name}.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Save to log file only for test split
    if split_name == "test":
        output_file = f"results/embedding_training_{args.data_name}_{args.run_name}.log"
        max_key_length = max(len(key) for key in avg_results.keys())

        with open(output_file, "a") as f:
            f.write("\n\nResults\n")
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Time: {time_str}\n")
            for key, value in vars(args).items():
                f.write(f"{key:<26} : {value}\n")
            f.write("---\n")
            for key, value in avg_results.items():
                f.write(f"{key.ljust(max_key_length)} : {value:.4f}\n")

            # core metrics (tab-separated)
            metrics_order = ["hit@5", "ndcg@5", "hit@10", "ndcg@10"]
            f.write("\t".join([f"{avg_results[k]:.4f}" for k in metrics_order]) + "\n")

            # warm / cold metrics
            if warm_item_set is not None:
                wc_keys = ["warm_hit@5", "warm_ndcg@5", "warm_hit@10", "warm_ndcg@10",
                           "cold_hit@5", "cold_ndcg@5", "cold_hit@10", "cold_ndcg@10"]
                f.write("warm/cold:\t" +
                        "\t".join([f"{avg_results.get(k, 0.0):.4f}" for k in wc_keys]) + "\n")

            # novelty metrics
            if use_novelty:
                nov_keys = ["novelty@5", "novelty@10",
                            "novelty_hit@5", "novelty_ndcg@5",
                            "novelty_hit@10", "novelty_ndcg@10"]
                f.write("novelty:\t" +
                        "\t".join([f"{avg_results.get(k, 0.0):.4f}" for k in nov_keys]) + "\n")

        print(f"Log metrics saved to {output_file}")

        output_file = f"results/total_results.log"
        with open(output_file, "a") as f:
            f.write(f"Model: {args.model_name} {args.cf_model} {args.fusion_method}\n")
            f.write(f"run_name: {args.run_name}\n")
            f.write(f"seed: {args.seed}\n")
            f.write(f"Data: {args.data_name} \n")
            metrics_order = ["hit@5", "ndcg@5", "hit@10", "ndcg@10"]
            f.write("\t".join([f"{avg_results[k]:.4f}" for k in metrics_order]) + "\n")
            f.write("--------------------------------\n")

    return avg_results



if __name__ == "__main__":
    args = parse_arguments()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    encoder_dir = "encoder_1"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load the item embeddings
    item_embedding_file = f"data_emb/{args.data_name}_{args.item_target_names_str}_{args.token_limit}_{args.item_model_name}_{args.embedding_model_name_dir}_emb.pt"
    print("item_embedding_file:", item_embedding_file)
    item_embeddings = torch.load(item_embedding_file).cpu()


    train_cf_user_embs, val_cf_user_embs, test_cf_user_embs, cf_item_embs = load_cf_embeddings(args)
    print("train_cf_user_embs.shape:", train_cf_user_embs.shape)
    print("cf_item_embs.shape:", cf_item_embs.shape)
    user_num = len(train_cf_user_embs)
    item_num = len(cf_item_embs)
    item_hit_label = torch.ones(len(item_embeddings)+1) * 0.5


    args.llm_emb_dim = item_embeddings.shape[1]
    args.cf_emb_dim = cf_item_embs.shape[1]
    model = FusionModel(
        args=args,
        max_user_id=user_num,
        max_item_id=item_num,
        llm_emb_dim=args.llm_emb_dim,
        cf_emb_dim=args.cf_emb_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        llm_item_embs=item_embeddings,
        cf_item_embs=cf_item_embs,
        item_hit_label=item_hit_label.unsqueeze(1),
    )

    # load pos_item_ids and neg_item_sets
    data_file = f"data/{args.data_name}/sequential_data.txt"
    train_user_seq_data = {}
    val_user_seq_data = {}
    test_user_seq_data = {}
    train_pos_item_ids = []
    val_pos_item_ids = []
    test_pos_item_ids = []
    test_user_target = {}
    train_user_target = {}
    val_user_target = {}

    # select random negatives except for history items
    neg_item_sets = []
    total_items = set(range(1, item_embeddings.shape[0]))
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]

            part_set = set(parts)
            neg_candidate_items = total_items - part_set
            neg_items = random.sample(list(neg_candidate_items), args.num_neg_samples)
            neg_item_sets.append(neg_items)

            train_user_seq_data[user_id] = parts[:-3]
            val_user_seq_data[user_id] = parts[:-2]
            test_user_seq_data[user_id] = parts[:-1]
            train_pos_item_ids.append(parts[-3])
            val_pos_item_ids.append(parts[-2])
            test_pos_item_ids.append(parts[-1])
            train_user_target[user_id] = parts[-3]
            val_user_target[user_id] = parts[-2]
            test_user_target[user_id] = parts[-1]
    # Compute item popularity from train set for warm/cold split and novelty metrics
    item_popularity, warm_item_set, total_train_interactions = compute_item_popularity(
        train_user_seq_data, train_user_target
    )

    # load the user embeddings

    
    train_user_embeddings = torch.zeros((len(train_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)
    val_user_embeddings = torch.zeros((len(val_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)
    test_user_embeddings = torch.zeros((len(test_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)

    
    train_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_item_transformer_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_emb.pt"
    val_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_item_transformer_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_emb.pt"
    test_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_item_transformer_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_emb.pt"
    
    if args.use_llm_emb:
        print("🔹 Use llm emb")
        train_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_pred_emb.pt"
        val_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_pred_emb.pt"
        test_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_pred_emb.pt"
        
    train_user_embeddings = torch.load(train_user_embedding_file).to("cpu")
    val_user_embeddings = torch.load(val_user_embedding_file).to("cpu")
    test_user_embeddings = torch.load(test_user_embedding_file).to("cpu")
    
    # add zero padding
    if (not args.use_llm_emb) and (args.run_name!="llm_emb"):
        train_user_embeddings = torch.cat([torch.zeros((1, train_user_embeddings.shape[1])), train_user_embeddings], dim=0)
        val_user_embeddings = torch.cat([torch.zeros((1, val_user_embeddings.shape[1])), val_user_embeddings], dim=0)
        test_user_embeddings = torch.cat([torch.zeros((1, test_user_embeddings.shape[1])), test_user_embeddings], dim=0)

    print("test user embedding file:", test_user_embedding_file)
    print("train_user_embeddings.shape:", train_user_embeddings.shape)
    print("val_user_embeddings.shape:", val_user_embeddings.shape)
    print("test_user_embeddings.shape:", test_user_embeddings.shape)
    print("original dtype:", train_user_embeddings.dtype)


    train_user_embs = train_user_embeddings
    val_user_embs = val_user_embeddings
    test_user_embs = test_user_embeddings
    llm_item_embs = item_embeddings * args.gamma
    cf_item_embs = cf_item_embs

    # print mean and std of train_user_embeddings
    print("mean of train_user_embeddings:", train_user_embs.mean())
    print("std of train_user_embeddings:", train_user_embs.std())
    print("mean of train_user_embeddings:", train_user_embs[10,:].mean())
    print("std of train_user_embeddings:", train_user_embs[10,:].std())

    # exit()

    
    # negative sampling for training
    data_file = f"data/{args.data_name}/negative_samples.txt"
    neg_item_sets = []
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]
            # randomly sample 10 negative items
            if len(parts) > args.num_neg_samples:
                parts = random.sample(parts, args.num_neg_samples)
            # parts = random.sample(parts, 30)
            neg_item_sets.append(parts)


    train_hit_user_label = torch.ones(len(train_user_embs)+1) * 0.5
    valid_hit_user_label = torch.ones(len(val_user_embs)+1) * 0.5
    test_hit_user_label = torch.ones(len(test_user_embs)+1) * 0.5

    print("train_user_label:", train_hit_user_label[2])


    print("Merged train_user_embs.shape:", train_user_embs.shape)
    print("Merged val_user_embs.shape:", val_user_embs.shape)
    print("Merged test_user_embs.shape:", test_user_embs.shape)

    train_neg_item_sets = neg_item_sets

    # Create DataLoader
    train_loader = get_dataloader(
        train_user_embs, train_cf_user_embs,
        llm_item_embs, cf_item_embs,
        train_pos_item_ids, train_neg_item_sets,
        train_hit_user_label,
        batch_size=args.train_batch_size)
    train_eval_loader = get_dataloader(
        train_user_embs, train_cf_user_embs,
        llm_item_embs, cf_item_embs,
        train_pos_item_ids, train_neg_item_sets,
        train_hit_user_label,
        batch_size=args.train_batch_size,
        shuffle=False
    )
    del train_user_embs, train_cf_user_embs
    gc.collect()

    valid_loader = get_dataloader(
        val_user_embs, val_cf_user_embs,
        llm_item_embs, cf_item_embs,
        val_pos_item_ids, neg_item_sets,
        valid_hit_user_label,
        batch_size=args.eval_batch_size, shuffle=False)
    del val_user_embs, val_cf_user_embs
    gc.collect()
    test_loader = get_dataloader(
        test_user_embs, test_cf_user_embs,
        llm_item_embs, cf_item_embs,
        test_pos_item_ids, neg_item_sets,
        test_hit_user_label,
        batch_size=args.eval_batch_size, shuffle=False)
    del test_user_embs, test_cf_user_embs
    gc.collect()


    # Train the model
    print("Training the model...")
    if args.epochs > 0 and not args.zero_shot:
        trained_model = train_rec(
            args, model, train_loader, valid_loader, epochs=args.epochs, device=str(device)
        )
    else:
        trained_model = model



# Evaluate on all splits
popularity_kwargs = dict(
    item_popularity=item_popularity,
    warm_item_set=warm_item_set,
    total_train_interactions=total_train_interactions,
)

if "train" in args.target_split:
    print("Evaluating on train split...")
    train_results = evaluate_model(
        trained_model, train_eval_loader, train_user_seq_data, train_user_target,
        device, "train", args, **popularity_kwargs
    )

if "val" in args.target_split:
    print("Evaluating on validation split...")
    valid_results = evaluate_model(
        trained_model, valid_loader, val_user_seq_data, val_user_target,
        device, "valid", args, **popularity_kwargs
    )

if "test" in args.target_split:
    print("Evaluating on test split...")
    test_results = evaluate_model(
        trained_model, test_loader, test_user_seq_data, test_user_target,
        device, "test", args, **popularity_kwargs
    )