import argparse
import json
import html
from tqdm import tqdm
import torch
import gc

from transformers import pipeline, AutoConfig
from sentence_transformers import SentenceTransformer

def get_item_meta_text(item_meta_dict):
    item_meta_text = ""
    # for key in ["title", "brand", "category", "description"]:
    for key in ["title", "brand", "category"]:
        if key in item_meta_dict:
            item_meta_text += f"{key}: {item_meta_dict[key]}\n"
    return item_meta_text

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

def get_title_review_text(user_seq_data, item_meta, user2item2_review):
    user_inputs = ["Unknown"]
    for user_id in range(1, len(user_seq_data)+1):
        history_iids = user_seq_data[user_id][-8:]
        history_text = ""
        for iid in history_iids:
            history_text += item_meta[iid] + "\n"
            history_text += user2item2_review[user_id][iid] + "\n"
        user_inputs.append(history_text)
    return user_inputs


def extract_embeddings(args, input_texts, llm_pipeline, output_file, max_length):
    if "sentence-transformers" in args.embedding_model_name.lower() or args.use_sentence_transformers:
        total_embeddings = llm_pipeline.encode(input_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True).cpu()
        torch.save(total_embeddings, output_file)
        del total_embeddings
        gc.collect()
        return
    try:
        hidden_size = llm_pipeline.model.config.hidden_size
    except:
        hidden_size = llm_pipeline.model.config.text_config.hidden_size
    
    total_embeddings = torch.zeros((len(input_texts), hidden_size), dtype=torch.float32)
    for i in tqdm(range(0, len(input_texts), args.batch_size)):
        with torch.no_grad():
            batch_text = input_texts[i: i + args.batch_size]
            batch_embeddings = llm_pipeline(batch_text, batch_size=args.batch_size, return_tensors="pt", max_length=max_length, truncation=True)
            for j, emb in enumerate(batch_embeddings):
                emb = emb[:, args.target_index, :]
                total_embeddings[i+j] = emb.squeeze(0)
                del emb
            gc.collect()

    # check the shape of the embeddings
    assert total_embeddings.size(0) == len(input_texts),\
        f"Total embeddings size: {total_embeddings.size(0)} != {len(input_texts)}"
    torch.save(total_embeddings, output_file)
    return


def process_item_features(args):
    """
    Processes item features to extract title embeddings for a specified dataset.

    Args:
        data_name (str): Name of the dataset (e.g., Luxury_Beauty, Video_Games).
        model_name (str): Name of the SentenceTransformer model.
        device (str): Device to use ('cuda:1' or 'cpu').

    Returns:
        None: Saves the processed embeddings as a .pt file.
    """
    data_name = args.data_name
    model_name = args.embedding_model_name
    device = args.device

    # File paths
    model_dir_name = model_name.split("/")[-1]

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

    # load item meta json
    item_meta_file = f"data/{args.data_name}/meta_text_fix.json"
    with open(item_meta_file, 'r') as f:
        item_meta = json.load(f)
    item_meta = {int(k): get_item_meta_text(v) for k, v in item_meta.items()}

    # load user preference json
    train_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_train_user_preference.json"
    valid_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_valid_user_preference.json"
    test_user_preference_file = f"data_processed/{args.data_name}_{args.model_name}_test_user_preference.json"
    with open(train_user_preference_file, 'r') as f:
        train_user_preference = json.load(f)
    train_user_preference = {int(k): v for k, v in train_user_preference.items()}
    with open(valid_user_preference_file, 'r') as f:
        val_user_preference = json.load(f)
    val_user_preference = {int(k): v for k, v in val_user_preference.items()}
    with open(test_user_preference_file, 'r') as f:
        test_user_preference = json.load(f)
    test_user_preference = {int(k): v for k, v in test_user_preference.items()}


    if args.target == "title_review":
        # load user2review.json
        user2review_file = f"data/{args.data_name}/user2reviews.json"
        with open(user2review_file, 'r') as f:
            user2review = json.load(f)
        user2review = {int(k): v for k, v in user2review.items()}
        user2item2review = {}
        for user_id, reviews in user2review.items():
            user2item2review[user_id] = {}
            for review in reviews:
                iid = int(review["item_id"])
                user2item2review[user_id][iid]=review["text"]
        # get user text
        train_user_text = get_title_review_text(train_user_seq_data, item_meta, user2item2review)
        val_user_text = get_title_review_text(val_user_seq_data, item_meta, user2item2review)
        test_user_text = get_title_review_text(test_user_seq_data, item_meta, user2item2review)
    
    else:
        train_user_text = get_user_text(args, train_user_seq_data, train_user_preference, item_meta)
        val_user_text = get_user_text(args, val_user_seq_data, val_user_preference, item_meta)
        test_user_text = get_user_text(args, test_user_seq_data, test_user_preference, item_meta)

    for idx in [30, 40, 50]:
        print(test_user_text[idx])
        print("="*50)

    from transformers import AutoModel, AutoTokenizer

    if "llama" in model_name.lower():
        llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        llm_pipeline.tokenizer.add_special_tokens({'pad_token': llm_pipeline.tokenizer.eos_token})
        llm_pipeline.tokenizer.padding_side = "left"
        max_length = args.token_limit
    elif "sentence-transformers" in model_name.lower() or args.use_sentence_transformers:
        llm_pipeline = SentenceTransformer(args.embedding_model_name, trust_remote_code=True)
        max_length = args.token_limit
        # llm_pipeline.max_seq_length = args.token_limit//2
    else:
        llm_pipeline = pipeline('feature-extraction', model=args.embedding_model_name, device=0)
        max_length = args.token_limit
        llm_pipeline.tokenizer.padding_side = "left"
        if "gemma" in model_name.lower():
            config = AutoConfig.from_pretrained(args.embedding_model_name)
            config.text_config.use_cache = False
            llm_pipeline = pipeline(
                    'feature-extraction', 
                    model=args.embedding_model_name, 
                    config=config,
                    device=0,
                )

        # llm_pipeline.tokenizer.padding_side = "left"

    extract_embeddings(
            args, train_user_text, llm_pipeline, 
            f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_pred_emb.pt", 
            max_length,
        )
    extract_embeddings(
            args, val_user_text, llm_pipeline, 
            f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_pred_emb.pt", 
            max_length,
        )
    extract_embeddings(
            args, test_user_text, llm_pipeline, 
            f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_pred_emb.pt", 
            max_length,
        )

    
def main():
    """
    Main function to process item features for a given dataset.
    """
    parser = argparse.ArgumentParser(description="Extract item title embeddings for a given dataset.")
    parser.add_argument(
        "--data_name", type=str, required=True, help="Name of the dataset (e.g., Luxury_Beauty, Video_Games)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for processing (default: cuda:1)"
    )
    parser.add_argument("--run_name", type=str, default="pref", help="Name of the dataset to use.")
    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct", help="Name of the SentenceTransformer model.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the SentenceTransformer model.")
    parser.add_argument("--target", type=str, default="title", help="Target feature to extract embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing item features.")
    parser.add_argument("--target_index", type=int, default=0, help="Target index for extracting embeddings.")
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    # parser.add_argument("--split", type=str, default="train", help="Split to process (train/test).")
    parser.add_argument("--token_limit", type=int, default=1024, help="Token limit for the model.")
    parser.add_argument("--use_sentence_transformers", action="store_true", help="Use SentenceTransformers for embedding extraction.")

    args = parser.parse_args()

    args.target_list = args.target.split("_")
    args.embedding_model_name_dir = args.embedding_model_name.split("/")[-1]

    print(f"Processing {args.data_name} dataset with target: {args.target_list}")

    # Process the dataset
    process_item_features(args)

if __name__ == "__main__":
    main()