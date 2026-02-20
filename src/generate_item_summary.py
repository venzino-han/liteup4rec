import json
import argparse

import random
from collections import defaultdict

from llm_generator import LLMGenerator

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

ITEM_SUMMARY_GENERATION_SYSTEM_PROMPT = """
You are an intelligent assistant that analyzes user preferences to generate personalized item recommendations.
""".strip()

ITEM_SUMMARY_GENERATION_PROMPT = """
### Task:
Analyze the provided metadata and reviews to determine what kinds of users are most likely to prefer the target item.
Your response should start with:  
**"Users who prefer [common themes/preferences] would find this item suitable."**  
Incorporate patterns from the reviews, such as favored features, usage scenarios, or functional benefits.
Avoid generic statements and ensure your description is grounded in the review content.

### Target Item Metadata:
- **Target Item Title**: {item_title}
{brand_line}- **Original Description**: {description}

### Requirements:
- Begin with the sentence: "Users who prefer ..."
- Respond with a concise and information-dense single paragraph, strictly limited to {max_words} words. Avoid redundancy and ensure clarity.

### Response:
""".strip()

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
    output_file = f"data_processed/{data_name}_{model_dir_name}_item_{args.run_name}.json"

    print(output_file)
    # Load item features
    with open(input_file, 'r') as f:
        item_metadata = json.load(f)
    item_metadata = {int(k): v for k, v in item_metadata.items()}

    reviews_file = f"data/{data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    # collect reviews for each item's previous interaction
    item_2_pre_reviews = defaultdict(list)
    item_2_pre_items = defaultdict(list)
    if args.use_previous_item_reviews:
        for user_id, reviews in user2reviews.items():
            review_text = ""
            pre_item = -1
            for review in reviews[:-1]: # only use train reviews
                item_id = int(review["item_id"])

                if pre_item != -1 and review_text != "":
                    # limit review text words to args.word_limit
                    item_2_pre_items[item_id].append(pre_item)
                    item_2_pre_reviews[item_id].append(review_text)

                review_text = " ".join(review["text"].split()[:args.word_limit])
                pre_item = item_id
    else:
        for user_id, reviews in user2reviews.items():
            for review in reviews:
                item_id = int(review["item_id"])
                item_2_pre_items[item_id].append(item_id)
                item_2_pre_reviews[item_id].append(review["text"])

    # for r in reviews:
    #     print(r)
    # exit()

    # min max number of reviews
    min_review_count = 10000
    max_review_count = 0
    for item_id, reviews in item_2_pre_reviews.items():
        review_count = len(reviews)
        if review_count < min_review_count:
            min_review_count = review_count
        if review_count > max_review_count:
            max_review_count = review_count
    print(f"min review count: {min_review_count}")
    print(f"max review count: {max_review_count}")

    # LLM 기반 요약 생성
    print("Generating item summaries using LLM...")
    count = 0
    prompts = []
    item_ids = []
    for item_id, item_data in item_metadata.items():
        item_title, brand, description = item_data["title"], item_data["brand"], item_data["description"]
        description = remove_html_tags(description)
        description = " ".join(description.split()[:args.word_limit])
        if args.data_name == "yelp":
            brand_line = ""
        else:
            brand_line = f"**Brand**: {brand}"

        # Prepare prompt with item features
        prompt = ITEM_SUMMARY_GENERATION_PROMPT.format(
            item_title=item_title,
            brand_line=brand_line,
            description=description,
            max_words=args.max_words,
        )
        prompts.append(prompt)
        item_ids.append(item_id)

    print(count)
    print("prompt sample:")
    for i in [10, 20, 30]:
        print(f"Prompt {i}:")
        print(prompts[i])
        print("------\n")

    # Generate responses using vLLM

    llm_generator = LLMGenerator(args)
    responses, token_nums = llm_generator.generate_response(ITEM_SUMMARY_GENERATION_SYSTEM_PROMPT, prompts)

    print("Response sample:")
    for i in [10, 20, 30]:
        print(f"Response {i}:")
        print(responses[i])
        print("------\n")

    # Update item features with generated summaries
    item_summary = {}
    for item_id, summary in zip(item_ids, responses):
        item_summary[item_id] = summary.strip()
    
    # Save item summaries to file
    with open(output_file, 'w') as f:
        json.dump(item_summary, f, indent=4)

if __name__ == "__main__":
    random.seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Run inference and generate test results.")
    parser.add_argument("--run_name", type=str, default="basic", help="Name of the dataset to use.")
    parser.add_argument("--data_name", type=str, default="beauty", help="Name of the dataset to use.")
    parser.add_argument("--check_interval", type=int, default=100, help="Number of samples to use for testing.")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the model to use.")    
    parser.add_argument("--max_input_tokens", type=int, default=1024*6, help="Max tokens for sampling.")
    parser.add_argument("--max_output_tokens", type=int, default=1024, help="Max tokens for sampling.")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for sampling.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for sampling.")

    parser.add_argument("--use_reviews", action="store_true", help="Use reviews for item summary generation.")
    parser.add_argument("--word_limit", type=int, default=128, help="")
    parser.add_argument("--review_limit", type=int, default=16, help="")
    parser.add_argument("--max_words", type=int, default=128, help="")
    parser.add_argument("--use_previous_item_reviews", action="store_true", help="Use previous item reviews for item summary generation.")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing item features.")
    parser.add_argument("--use_deep_infra", action="store_true", help="Use Deep Infra for inference.")

    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")

    args = parser.parse_args()

    process_item_features(args)