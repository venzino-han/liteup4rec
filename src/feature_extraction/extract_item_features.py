""" extract item attributes from item title and review text """

import pandas as pd
import json
import ollama
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.common import change_special_chars, process_batch_responses

ITEM_SUMMARY_SYSTEM_PROMPT = """
You are an AI assistant analyzing item titles, descriptions, and user reviews to extract attributes and key features for search systems.
"""

ITEM_ATTRIBUTE_FEATURE_PROMPT = """
Given the item title, description, and user reviews, identify and list the key attributes and features of the item.
Prioritize listing unique and differentiating factors that distinguish this item from others.
Attributes are specific properties, and features are notable aspects or characteristics.

### Task:
Given the provided item title, description, and user reviews, extract a list of key attributes and features.
Ensure that the attributes and features are relevant to the item and useful for search and filtering.
Prioritize unique and distinguishing aspects.

### Item Data:
{item_info}

### Output Format:
Return a list of attributes and key features, separated by newlines.

- [Unique Attribute/Feature 1]
- [Unique Attribute/Feature 2]
- [Other Attribute/Feature 3]
- [Other Attribute/Feature 4]
- ...

Ensure each attribute and feature is concise and relevant. Don't add any additional comments.

### Output Example:
{output_example}
"""

OUTPUT_EXAMPLE_TOYS = """
- Interactive voice commands
- Programmable dance routines
- Durable, child-safe material
- Educational STEM challenges
- Long battery life for extended play
"""

OUTPUT_EXAMPLE_BEAUTY = """
- Hypoallergenic, fragrance-free formula
- Patented micro-delivery system
- Visible results in 7 days
- Cruelty-free and vegan certified
- SPF 50+ broad spectrum protection
"""

OUTPUT_EXAMPLE_SPORTS = """
- Ultra-lightweight carbon fiber frame
- Customizable gear ratios
- Built-in GPS and performance tracker
- Weather-resistant, breathable fabric
- Impact-resistant, shatterproof lens
"""

EXAMPLE_MAP = {
    "toys": OUTPUT_EXAMPLE_TOYS,
    "beauty": OUTPUT_EXAMPLE_BEAUTY,
    "sports": OUTPUT_EXAMPLE_SPORTS,
}

def interactions_to_text(interactions, max_len=8):
    if len(interactions) > max_len:
        interactions = interactions[-max_len:]
    text = ""
    for interaction in interactions:
        text += f"Rating: {interaction['rating']}, User Review: {interaction['review']}\n"
    return text

def get_item_feature_prompt(item_title, item_description, interactions, examples, prompt_format):
    interaction_text = interactions_to_text(interactions)
    item_info = f"**Title:** {item_title}\n**Description:** {item_description}\n**User Reviews:**\n{interaction_text}"
    prompt = prompt_format.format(
        item_info=item_info, 
        output_example=examples
        )
    return prompt


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="Summarize product descriptions using Ollama")
    parser.add_argument("--data_name", type=str, default="Luxury_Beauty", help="Name of the dataset")
    parser.add_argument("--model_name", type=str, default="gemma2:2b-instruct-fp16", help="Model name for Ollama or OpenAI")
    args = parser.parse_args()

    assert args.data_name in EXAMPLE_MAP, f"Data name {args.data_name} not supported"

    file_name = f"data/{args.data_name}_item_meta_dict.json"
    with open(file_name, 'r') as f:
        meta_data = json.load(f)

    file_name = f"data/{args.data_name}_item_old2new_id_dict.json"
    with open(file_name, 'r') as f:
        old2new_id = json.load(f)
    new2old_id = {v: k for k, v in old2new_id.items()}

    file_name = f"data/{args.data_name}_item_interaction_dict.json"
    with open(file_name, 'r') as f:
        interaction_data = json.load(f)

    print("Start processing data")
    print(f"Dataset: {args.data_name}")
    print(f"Model: {args.model_name}")

    DEFAULT_VALUE = {"title": "No Title", "description": "No Description"}
    new_file_name = f"data_processed/{args.data_name}_{args.model_name}_item_features.json"
    new_data = {}
    prompts = []
    titles = []
    descriptions = []
    item_ids = []
    for i, (item_id, interactions) in enumerate(tqdm(interaction_data.items())):
        item_id = int(item_id)
        old_id = new2old_id[item_id]
        title = meta_data.get(old_id, DEFAULT_VALUE)["title"]
        desc = meta_data.get(old_id, DEFAULT_VALUE)["description"]
        titles.append(title)
        descriptions.append(desc)
        item_ids.append(item_id)

        examples = EXAMPLE_MAP.get(args.data_name)
        prompt = get_item_feature_prompt(
            item_title=title,
            item_description=desc,
            interactions=interactions,
            examples=examples,
            prompt_format=ITEM_ATTRIBUTE_FEATURE_PROMPT
        )
        prompt = change_special_chars(prompt)
        prompts.append((prompt, ITEM_SUMMARY_SYSTEM_PROMPT))

        print(prompt)
        exit()

    # batch processing
    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[i:i+batch_size]
        results = process_batch_responses(client, prompt_batch, model_name=args.model_name)
        for j in range(len(results)):
            item_id = item_ids[i+j]
            new_data[item_id] = {
                "title": titles[i+j],
                "description": descriptions[i+j],
                "features": results[j]
            }
            with open(f"temp/item_features_{args.data_name}_{args.model_name}.txt", 'a') as f:
                f.write(f"Prompt:\n{prompt_batch[j][0]}\n\nResponse:\n{results[j]}\n\n")
                f.write("-"*100+"\n")

    with open(new_file_name, 'w') as f:
        json.dump(new_data, f, indent=4)




