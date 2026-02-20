import json
import html
import argparse
import re
from tqdm import tqdm
import torch
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, Gemma3ForCausalLM, Gemma3TextModel
from sentence_transformers import SentenceTransformer

from extract_user_emb import (
    get_item_meta_text,
)


def apply_chat_template(tokenizer, system_prompt, prompt):
    prompt = [
        {"role": "system", "content": system_prompt,},
        {"role": "user", "content": prompt,},
    ]
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)


USER_PREFERENCE_SYSTEM_PROMPT = """
You are an intelligent assistant that analyzes a user's recent preferences based on their chronological purchase history and associated reviews.
Your goal is to infer the user's current interests, priorities, and behavioral patterns by identifying consistent themes or shifts in their preferences.
Focus on highlighting the most recent signals of interest, usage needs, and valued attributes derived from their review content.
Avoid speculation, and base your analysis strictly on the provided data.
""".strip()

USER_PREFERENCE_PROMPT = """
Based on the user's chronological purchase history and their written reviews, analyze and summarize their most recent preferences and interests.

User's purchase history:
{user_purchase_review_history}
""".strip()

def generate_user_preferences(args, split="train"):
    """
    Generate user preferences using LLM based on purchase history and reviews.
    """
    data_name = args.data_name
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_dir_name = model_name.split("/")[-1]
    input_file = f"data/{data_name}/meta_text_fix.json"
    output_file = f"data_emb/{data_name}_{model_dir_name}_{split}_user_emb.pt"

    print(f"Generating user preferences for {split} split...")
    print(f"Output file: {output_file}")

    # Load item features
    with open(input_file, 'r') as f:
        item_metadata = json.load(f)
    item_metadata = {int(k): v for k, v in item_metadata.items()}

    reviews_file = f"data/{data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    uid_2_history_text = {}
    if split == "train":
        index = -3
    elif split == "valid":
        index = -2
    elif split == "test":
        index = -1
    
    for user_id, reviews in user2reviews.items():
        history_text = ""
        target_timestamp = int(reviews[index]["timestamp"])
        for review in reviews[index-args.history_limit:index]:
            item_id = int(review["item_id"])
            item_title = item_metadata[item_id]["title"]
            brand = item_metadata[item_id]["brand"]
            category = item_metadata[item_id]["category"]
            review_text = review["text"]
            # limit review text words
            if len(review_text.split()) > args.word_limit:
                review_text = " ".join(review_text.split()[:args.word_limit])
            date = review["date"]
            timestamp = int(review["timestamp"])
            # only include reviews that are in the last 14 days
            if target_timestamp - timestamp > 14 * 24 * 60 * 60:
                continue
            history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\nReview:\n{review_text}\n\n"

        # if no reviews in the last 14 days, include the last review
        if len(history_text) < 1:
            history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\nReview:\n{review_text}\n\n"
            
        uid_2_history_text[int(user_id)] = history_text

    # Generate user preferences using LLM
    prompts = []
    user_ids = []
    for user_id, user_history_text in uid_2_history_text.items():
        prompt = USER_PREFERENCE_PROMPT.format(
            user_purchase_review_history=user_history_text,
        )
        if "gemma" in model_name.lower():
            prompt = apply_chat_template(tokenizer, USER_PREFERENCE_SYSTEM_PROMPT, prompt)
        prompts.append(prompt)
        user_ids.append(user_id)
    
    return prompts, user_ids


def get_user_text(args, user_seq_data, user_preference, item_meta):
    user_inputs = ["Unknown"]
    for user_id in range(1, len(user_seq_data)+1):
        history_iids = user_seq_data[user_id]
        recent_iid = history_iids[-1]
        history_text = user_preference[user_id]
        if args.add_item_meta:
            history_text = "This user recently purchased:\n" + item_meta[recent_iid] + "\n" + \
                "Here is the user's preference:\n" + history_text
        user_inputs.append(history_text)
    return user_inputs


def extract_embeddings(args, prompts, tokenizer, model, output_file, ):
    """
    Extract embeddings from the model, with option to save multiple last tokens.
    """
    if "gemma" in args.model_name.lower():
        # tokenize prompts
        tokenizer.padding_side = "left"
        tokenized_prompts = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=args.token_limit)
        model.to(args.device)
        model.eval()
        # batchfy tokenized_prompts
        total_embeddings = []
        for i in tqdm(range(0, len(prompts), args.batch_size), total=len(prompts)//args.batch_size+1, desc="Extracting embeddings"):
            batch_prompts = {
                "input_ids": tokenized_prompts["input_ids"][i:i+args.batch_size],
                "attention_mask": tokenized_prompts["attention_mask"][i:i+args.batch_size],
            }
            batch_prompts = {k: v.to(args.device) for k, v in batch_prompts.items()}
            with torch.no_grad():
                outputs = model(**batch_prompts)
                embeddings = outputs.last_hidden_state[:, -1, :].cpu()
            total_embeddings.append(embeddings)
        total_embeddings = torch.cat(total_embeddings, dim=0)
    else:
        total_embeddings = model.encode(prompts, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True).cpu()
        
    # save embeddings
    print(f"Saving embeddings to {output_file}")
    print(f"Embeddings shape: {total_embeddings.shape}")
    torch.save(total_embeddings, output_file)
    return


def process_user_preference_embeddings(args):
    """
    Process user preference embeddings for a specified dataset.
    """
    data_name = args.data_name
    model_name = args.model_name
    device = args.device

    # File paths
    model_dir_name = model_name.split("/")[-1]

    # Load sequential data
    data_file = f"data/{args.data_name}/sequential_data.txt"
    train_user_seq_data = {}
    val_user_seq_data = {}
    test_user_seq_data = {}
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]
            train_user_seq_data[user_id] = parts[:-3]
            val_user_seq_data[user_id] = parts[:-2]
            test_user_seq_data[user_id] = parts[:-1]

    # Load item metadata
    item_meta_file = f"data/{args.data_name}/meta_text_fix.json"
    with open(item_meta_file, 'r') as f:
        item_meta = json.load(f)
    item_meta = {int(k): get_item_meta_text(v) for k, v in item_meta.items()}

    # Generate or load user preferences
    print("Generating user preferences...")
    train_user_prompts, train_user_ids = generate_user_preferences(args, split="train")
    val_user_prompts, val_user_ids = generate_user_preferences(args, split="valid")
    test_user_prompts, test_user_ids = generate_user_preferences(args, split="test")

    # Print sample texts
    for idx in [30, 40, 50]:
        print(f"Sample user prompt {idx}:")
        print(test_user_prompts[idx])
        print("="*50)

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "gemma" in model_name.lower():
        # only use Gemma3TextModel from Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).model
        print(f"Model dtype: {model.dtype}")
    else:
        model = SentenceTransformer(model_name)
    
    
    train_output_file = f"data_emb/{args.data_name}_{args.model_name_dir}_train_user_emb.pt"
    val_output_file = f"data_emb/{args.data_name}_{args.model_name_dir}_val_user_emb.pt"
    test_output_file = f"data_emb/{args.data_name}_{args.model_name_dir}_test_user_emb.pt"

    # Extract embeddings for each split
    print("Extracting train embeddings...")
    extract_embeddings(
        args, train_user_prompts, tokenizer, model, train_output_file,
    )
    print("Extracting val embeddings...")
    extract_embeddings(
        args, val_user_prompts, tokenizer, model, val_output_file,
    )
    print("Extracting test embeddings...")
    extract_embeddings(
        args, test_user_prompts, tokenizer, model, test_output_file,
    )


def main():
    """
    Main function to process user preference embeddings for a given dataset.
    """
    parser = argparse.ArgumentParser(description="Extract user preference embeddings for a given dataset.")
    parser.add_argument(
        "--data_name", type=str, required=True, help="Name of the dataset (e.g., beauty, video_games)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for processing (default: cuda:0)"
    )
    parser.add_argument("--run_name", type=str, default="pref", help="Name of the run.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it", help="Name of the embedding model.")

    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for processing.")
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    parser.add_argument("--token_limit", type=int, default=1024*2, help="Token limit for the model.")

    parser.add_argument("--use_sentence_transformers", action="store_true", help="Use SentenceTransformers for embedding extraction.")
    parser.add_argument("--last_k_tokens", type=int, default=1, help="Number of last tokens to save (default: 1 for last token only).")

    parser.add_argument("--word_limit", type=int, default=128, help="Word limit for review text.")
    parser.add_argument("--history_limit", type=int, default=8, help="Number of history items to consider.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")

    args = parser.parse_args()
    args.model_name_dir = args.model_name.split("/")[-1]

    print(f"Processing {args.data_name} dataset")
    print(f"Model: {args.model_name}")
    print(f"Last k tokens: {args.last_k_tokens}")

    # Process the dataset
    process_user_preference_embeddings(args)


if __name__ == "__main__":
    main() 