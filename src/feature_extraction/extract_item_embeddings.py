from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import load_json

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model_name', type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument('--data_name', type=str, default="Luxury_Beauty")
    parser.add_argument('--summary_model_name', type=str, default="llama3.1:70b-instruct-q8_0")
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()

    model = SentenceTransformer(args.embedding_model_name, device=args.device)
    # Luxury_Beauty_gpt-4o-mini_1000_generated_item_features.json

    """ load item data"""
    item_metadata_file = f"data_processed/{args.data_name}_{args.summary_model_name}_item_summary.json"
    item_metadata = load_json(item_metadata_file)
    item_metadata = {int(k): v for k, v in item_metadata.items()}

    max_item_id = max(item_metadata.keys())

    print("Max item id: ", max_item_id)
    feature_embeddings = np.zeros((max_item_id + 1, 1024))
        
    """ encode item features """
    for i, (item_id, target_item_metadata) in enumerate(tqdm(item_metadata.items())):
        item_feature_text = target_item_metadata['summary'].strip()
        # attribute = target_item_metadata['attribute']
        title = target_item_metadata['title']

        item_feature_embedding = model.encode(item_feature_text, device=args.device) # (1024,) ndarray
        feature_embeddings[item_id] = item_feature_embedding

    print("Item embeddings shape: ", feature_embeddings.shape)
    # save item embeddings
    np.save(f"data_processed/{args.data_name}_{args.summary_model_name}_item_embeddings.npy", feature_embeddings)