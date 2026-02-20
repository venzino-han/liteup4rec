import json
import html
import argparse

import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm
import random
from collections import defaultdict
from transformers import AutoTokenizer
from llm_generator import LLMGenerator

from utils.common import (
    get_vllm_model,
    get_sampling_params,
    generate_response_with_vllm,
    )

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


USER_PREFERENCE_SYSTEM_PROMPT = """
You are an intelligent assistant that analyzes a user's recent preferences based on their chronological purchase history and associated reviews.
Your goal is to infer the user's current interests, priorities, and behavioral patterns by identifying consistent themes or shifts in their preferences.
Focus on highlighting the most recent signals of interest, usage needs, and valued attributes derived from their review content.
Avoid speculation, and base your analysis strictly on the provided data.
""".strip()


USER_PREFERENCE_PROMPT = """
### Task:
Based on the user's chronological purchase history and their written reviews, analyze and summarize their most recent preferences and interests.
Focus on identifying recent trends, valued features, and notable shifts in their behavior or expectations.

### Input:
{user_purchase_history}

### Requirements:
- Write a single coherent paragraph (max 512 words) summarizing the user’s **current or most recent** preferences.
- Emphasize patterns from the latest reviews more than older ones.
- Avoid quoting reviews directly; extract and paraphrase meaningful signals.
- Do not include personal opinions or assumptions beyond what is stated in the reviews.

### Response:
""".strip()



USER_PREFERENCE_WO_REVIEWS_SYSTEM_PROMPT = """
You are an intelligent assistant that analyzes a user's recent preferences based on their chronological purchase history.
Your goal is to infer the user's current interests, priorities, and behavioral patterns by identifying consistent themes or shifts in their preferences.
Avoid speculation, and base your analysis strictly on the provided data.
""".strip()


USER_PREFERENCE_WO_REVIEWS_PROMPT = """
### Task:
Based on the user's chronological purchase history, analyze and summarize their most recent preferences and interests.
Focus on identifying recent trends, valued features, and notable shifts in their behavior or expectations.

### Input:
{user_purchase_history}

### Requirements:
- Write a single coherent paragraph (max 512 words) summarizing the user’s **current or most recent** preferences.
- Emphasize patterns from the latest more than older ones.
- Avoid quoting all data directly; extract and paraphrase meaningful signals.
- Do not include personal opinions or assumptions beyond what is stated in the purchase history.

### Response:
""".strip()



# USER_PREFERENCE_PROMPT = """
# Based on the user's chronological purchase history and their written reviews, analyze and summarize their most recent preferences and interests.
# Focus on identifying recent trends, valued features, and notable shifts in their behavior or expectations.

# ### User Purchase Review History:
# {user_purchase_review_history}
# """.strip()

import re

def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def process_item_features(args):
    """
    Processes item features to extract title embeddings for a specified dataset,
    and optionally generates new item summaries using an LLM.
    """
    # 기존 코드: 데이터 로드 및 경로 설정
    data_name = args.data_name
    model_name = args.model_name

    model_dir_name = model_name.split("/")[-1]
    input_file = f"data/{data_name}/meta_text_fix.json"
    output_file = f"data_processed/{data_name}_{model_dir_name}_{args.split}_user_preference.json"

    print(output_file)
    # Load item features
    with open(input_file, 'r') as f:
        item_metadata = json.load(f)
    item_metadata = {int(k): v for k, v in item_metadata.items()}

    reviews_file = f"data/{data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    uid_2_history_text = {}
    if args.split == "train":
        index = -3
    elif args.split == "valid":
        index = -2
    elif args.split == "test":
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
            # only include reviews that are in the last 30 days
            if target_timestamp - timestamp > 14 * 24 * 60 * 60:
                continue

            if args.use_reviews:
                if args.data_name == "yelp":
                    history_text += f"Date: {date}\nItem Title: {item_title}\nCategory: {category}\nReview:\n{review_text}\n\n"
                else:
                    history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\nReview:\n{review_text}\n\n"
            else:
                if args.data_name == "yelp":
                    history_text += f"Date: {date}\nItem Title: {item_title}\nCategory: {category}\n\n"
                else:
                    history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\n\n"

        # if no reviews in the last 14 days, include the last review
        if len(history_text) < 1:
            if args.use_reviews:
                if args.data_name == "yelp":
                    history_text += f"Date: {date}\nItem Title: {item_title}\nCategory: {category}\nReview:\n{review_text}\n\n"
                else:
                    history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\nReview:\n{review_text}\n\n"
            else:
                if args.data_name == "yelp":
                    history_text += f"Date: {date}\nItem Title: {item_title}\nCategory: {category}\n\n"
                else:
                    history_text += f"Date: {date}\nItem Title: {item_title}\nBrand: {brand}\nCategory: {category}\n\n"
            
        uid_2_history_text[user_id] = history_text

    if args.use_reviews:
        SYSTEM_PROMPT = USER_PREFERENCE_SYSTEM_PROMPT
        PROMPT = USER_PREFERENCE_PROMPT
    else:
        SYSTEM_PROMPT = USER_PREFERENCE_WO_REVIEWS_SYSTEM_PROMPT
        PROMPT = USER_PREFERENCE_WO_REVIEWS_PROMPT

    # LLM 기반 요약 생성
    print("Generating item summaries using LLM...")
    prompts = []
    user_ids = []
    for user_id, user_history_text in tqdm(uid_2_history_text.items()):
        # Prepare prompt with item features
        prompt = PROMPT.format(
            user_purchase_history=user_history_text,
        )
        prompts.append(prompt)
        user_ids.append(user_id)

    print("prompt sample:")
    for i in range(3):
        print(prompts[i])

    # prompts = prompts[:10]

    llm_generator = LLMGenerator(args)
    responses, token_nums = llm_generator.generate_response(SYSTEM_PROMPT, prompts)

    print("Response sample:")
    for i in range(3):
        print(f"Response {i}:")
        print(responses[i])
        print("------\n")
    # # Generate responses using vLLM
    # model = get_vllm_model(args.model_name, args.num_gpus)
    # params = get_sampling_params(args)
    # responses = generate_response_with_vllm(model, prompts, params)


    # Update item features with generated summaries
    user_summary = {}
    for item_id, summary in zip(user_ids, responses):
        user_summary[item_id] = summary.strip()
    
    # Save item summaries to file
    with open(output_file, 'w') as f:
        json.dump(user_summary, f, indent=4)

if __name__ == "__main__":
    random.seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Run inference and generate test results.")
    parser.add_argument("--run_name", type=str, default="basic", help="Name of the dataset to use.")
    parser.add_argument("--data_name", type=str, default="beauty", help="Name of the dataset to use.")
    parser.add_argument("--check_interval", type=int, default=100, help="Number of samples to use for testing.")
    
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the model to use.")    
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for sampling.")
    parser.add_argument("--max_input_tokens", type=int, default=1024*6, help="Max tokens for sampling.")
    parser.add_argument("--max_output_tokens", type=int, default=1024, help="Max tokens for sampling.")
    parser.add_argument("--word_limit", type=int, default=128, help="")
    parser.add_argument("--history_limit", type=int, default=8, help="")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing item features.")
    parser.add_argument("--use_deep_infra", action="store_true", help="Use Deep Infra for inference.")
    parser.add_argument("--split", type=str, default="test", help="Split to use (train, valid, test).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--use_reviews", action="store_true", help="Use reviews in the user preference.")
    
    args = parser.parse_args()

    process_item_features(args)