
import random
from typing import List, Dict, Tuple
from nltk.tokenize import sent_tokenize

SYSTEM_PROMPT="""
You are an intelligent assistant designed to generate sequentially optimized search queries for recommending products based on the user's preferences and history.
Consider the user's past interactions to generate a query to seach a product that user will buy next.
""".strip()

# Define chat templates
GEMMA2_CHAT_TEMPLATE = """
<start_of_turn>user
{system_prompt}

Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response: <end_of_turn> <start_of_turn>
""".strip()

def generate_gemma2_chat(
    instruction: str, 
    input_text: str, 
    tokenizer, 
    system_prompt: str = "You are a helpful assistant.", 
) -> list:
    """
    Generates a chat conversation format for GEMMA2 using the provided instruction, input, and system prompt.
    Optionally applies the tokenizer to the generated chat template.

    Args:
        instruction (str): Instruction for the task.
        input_text (str): Input data for the task.
        tokenizer: Tokenizer object with `apply_chat_template` method.
        system_prompt (str): System prompt to set the assistant's behavior.
        tokenize (bool): Whether to apply tokenizer to the generated chat.

    Returns:
        list: A list of chat messages, tokenized if `tokenize` is True.
    """
    template_text = GEMMA2_CHAT_TEMPLATE.format(
        system_prompt=system_prompt,
        instruction=instruction,
        input_text=input_text,
    )
    return template_text

    # return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def generate_gemma2_response(response: str, tokenizer=None) -> list:
    """
    Generates a model's response in chat format for GEMMA2.
    Optionally applies the tokenizer to the response.

    Args:
        response (str): The model's response text.
        tokenizer: Tokenizer object with `apply_chat_template` method.
        tokenize (bool): Whether to apply tokenizer to the response.

    Returns:
        list: A list of response messages, tokenized if `tokenize` is True.
    """
    chat_response = f"""model {response}<end_of_turn>"""
    return chat_response

    # return tokenizer.apply_chat_template(chat_response, tokenize=False, add_generation_prompt=False)

LLAMA_INSTRUCTION_PROMPT="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:<|eot_id|>
"""

def generate_llama_instruction_prompt(instruction:str, input_text:str, system_prompt:str=SYSTEM_PROMPT)->str:
    return LLAMA_INSTRUCTION_PROMPT.format(system_prompt=system_prompt, instruction=instruction, input_text=input_text)

LLAMA_RESPONSE_PROMPT="""
<|start_header_id|>assistant<|end_header_id|>
{response}
<|eot_id|> 
"""
def generate_llama_response_prompt(response:str)->str:
    return LLAMA_RESPONSE_PROMPT.format(response=response).strip()

import re

def parse_llama_response_prompt(text: str) -> str:
    # Use regex to capture text between the assistant tags and the eot tag
    match = re.search(r"<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    elif "<|end_header_id|>" in text:
        return text.replace("<|start_header_id|>assistant<|end_header_id|>", "").replace("<|eot_id|>", "").replace("\n", "").strip()
    return text.strip().replace("\n", "")

def format_with_chat_template(
        tokenizer,
        system_prompt:str,
        user_prompt:str, 
        assistant_prompt:str,
        tokenize:bool=False,
        data_split:str="train",
    )->str:
    if data_split == "train":
        tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt},
            ], tokenize=False)


DEFAULT_ITEM_METADATA = {
    "title": "Unknown",
    "summary": "No summary available.",
    "description": "No description available."
}

def generate_target_user_history(
        args: Dict[str, object], 
        target_user_history: List[Dict[str, str]], 
        item_features: Dict[int, Dict[str, str]],
        item_info_limit: int = 48,
        split="train",
    ) -> str:
    target_user_history_list = []

    for i, interaction in enumerate(target_user_history):
        item_id = interaction["item_id"]
        item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
        target_user_history_text = f"{i+1}. `{item_metadata['title']}`"
        if args.use_query:
            query = item_metadata['master_query'].split('\n')[0]
            target_user_history_text += f"{i+1}. `{query}`"
        if args.use_summary:
            summary = item_metadata['summary'].strip()
            if len(summary.split()) > item_info_limit:
                summary = " ".join(summary.split()[:item_info_limit]) + "..."
            target_user_history_text += f"({summary})"
        if args.use_cf_summary:
            summary = item_metadata['item_cf_summary'].strip()
            if len(summary.split()) > item_info_limit:
                summary = " ".join(summary.split()[:item_info_limit]) + "..."
            target_user_history_text += f"({summary})"
        if args.use_rating:
            target_user_history_text += f", User Rating: {interaction['rating']}"
        if args.use_review:
            review = str(interaction['review']).strip()
            if len(review.split()) > item_info_limit:
                review = " ".join(review.split()[:item_info_limit]) + "..."
            target_user_history_text += f", User Review: `{review}`"
        if args.use_description:
            target_user_history_text += f" Product Description: {item_metadata['description']}"

        if args.use_similar_items:
            """ add top-k similar items """
            similar_items = item_metadata.get('similar_items', [])
            similar_items = similar_items[:args.similar_item_limit]
            similar_items_text = ", ".join(similar_items)
            similar_items_text = f"\nSimilar Items: {similar_items_text}"
            target_user_history_text += similar_items_text
        
        if args.use_next_items:
            """ add next item """
            next_item_id = item_metadata.get('next_items', [])
            next_item_metadata = item_features.get(next_item_id, DEFAULT_ITEM_METADATA)
            next_item_text = f"\nNext Item: `{next_item_metadata['title']}`"
            target_user_history_text += next_item_text

        target_user_history_text = f"{target_user_history_text}"
        target_user_history_list.append(target_user_history_text)

    if args.use_query:
        user_history_text = "".join(target_user_history_list)
        most_recent_query = item_features.get(target_user_history[-1]["item_id"], DEFAULT_ITEM_METADATA)["master_query"].split("\n")[0]
        return f"This user has searched the following queries in the past, in order:\n {user_history_text}\n"\
                + f"Note that the most recent query is `{most_recent_query}`."

    # Format the final output with an emphasis on the most recent interaction
    user_history_text = "\n".join(target_user_history_list)
    most_recent_title = item_features.get(target_user_history[-1]["item_id"], DEFAULT_ITEM_METADATA)['title']
    prompt = f"This user has interacted with the following items in the past, in order:\n{user_history_text}\n"
    if args.prompt_type=="recency-focused":
        prompt+=f"Note that the most recent interaction is with `{most_recent_title}`."
    if args.prompt_type=="llama_top5":
        prompt+=f"Note that the most recent interaction is with `{most_recent_title}`."
    return prompt

def generate_demo_text(
        args:Dict[str, object],
        demo_interacitons:List[List[Dict[str, str]]],
        demo_queries:List[str],
        item_features:Dict[int, Dict[str, str]],
        demo_interaction_limit:int=16,
        label_type:str="query",

    )->str:
    demo_text = "\n### Demonstrations:\nHere are some references from the similar users.\n"

    if label_type == "query+item-summary":
        for i, demo_query in enumerate(demo_queries):
            """ query and target item summary """
            demo_text += f"**Next Query:** {demo_query}\n"
            item_id = demo_interacitons[i][-1]["item_id"]
            item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
            demo_text += f"**Target Item Summary:** {item_metadata['summary']}\n"
        demo_text += "Demonstration is done.\n"
        return demo_text    
    elif label_type == "item-summary":
        for i, demo_query in enumerate(demo_queries):
            """ query and target item summary """
            item_id = demo_interacitons[i][-1]["item_id"]
            item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
            demo_text += f"**Target Item Summary:** {item_metadata['summary']}\n"
        demo_text += "Demonstration is done.\n"
        return demo_text
    elif label_type == "last+query":
        demo_text += f"**Demo User Last interaction:**\n"
        last_interaction = demo_interacitons[-1]
        item_id = last_interaction[-1]["item_id"]
        item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
        demo_text += f"`{item_metadata['title']}`"
        if args.demo_use_summary:
            demo_text += f"\nSummary: {item_metadata['summary'].strip()}"
        if args.demo_use_rating:
            demo_text += f"\nRating: {last_interaction[-1]['rating']}"
        if args.demo_use_review:
            demo_text += f"\nReview: {str(last_interaction[-1]['review']).strip()}"
        demo_text += f"\n**Next Query**: {demo_queries[-1]}\n"
        return demo_text
    elif label_type == "query-only":
        for i, demo_query in enumerate(demo_queries):
            demo_text += f"**Next Query {i+1}**: {demo_query}\n"
        demo_text += "Demonstration is done.\n"
        return demo_text

    for i, demo_query in enumerate(demo_queries):
        demo_text += f"<reference>\n**Demo User {i+1} History:**\n"
        for j, interaction in enumerate(demo_interacitons[i][-demo_interaction_limit:]):
            item_id = interaction["item_id"]
            item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
            demo_text += f"<interaction>\n`{item_metadata['title']}`"
            if args.demo_use_summary:
                demo_text += f"\nSummary: {item_metadata['summary'].strip()}"
            if args.demo_use_rating:
                demo_text += f"\nRating: {interaction['rating']}"
            if args.demo_use_review:
                demo_text += f"\nReview: {str(interaction['review']).strip()}"
            if args.demo_use_description:
                demo_text += f"\nDescription: {item_metadata['description']}"
            demo_text += "\n<\interaction>\n"
        if label_type == "query":
            demo_text += f"**Next Query**: {demo_query}\n"
        else:
            demo_text += f"**Next Title**: {demo_query}\n"
        demo_text += f"<\\reference>\n\n"
    return demo_text

import string

def get_cand_list_text(item_ids, item_features, args, spliter="\n") -> str:
    cand_list = []
    # random.shuffle(item_ids)
    
    for i, item_id in enumerate(item_ids):
        item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
        cand_list_temp = f"`{item_metadata.get('title', 'Unknown')}`"        
        cand_list.append(cand_list_temp)
    
    cand_list_text = spliter.join(cand_list)
    cand_list_text = f"{cand_list_text}"
    return cand_list_text

import json
def extract_query_from_generated_text(generated_text):
    # extract text betwee '{' and '}'
    try:
        start, end = generated_text.find('{'), generated_text.find('}')
        query_dict = json.loads(generated_text[start:end+1])
    except:
        query_dict = {"0":generated_text}
    return query_dict

def generate_pos_neg_query_pair_demo(userid_to_pos_neg_query_pair:Dict[int, Tuple[str, str]]) -> str:
    demo_text = "Here are some references from the similar users. **Successful Query** is the query that success to retrieve the target item, and **Failed Query** is the query that fails to retrieve the target item.\n"
    for i, (pos_query, neg_query) in enumerate(userid_to_pos_neg_query_pair.values()):
        demo_text += f"**Demo User {i+1}** **Successful Query:** `{pos_query}` **Failed Query:** `{neg_query}`\n"
    return demo_text


def generate_llm_rank_prompt(
        args: Dict[str, object], 
        target_user_history: List[int], 
        item_features: Dict[int, Dict[str, str]], 
        candidate_text_order: str,  # Candidates list in the form of a string
        cand_item_ids: List[int],  # List of candidate item IDs
        prompt_type: str = "sequential",  # Can be 'sequential', 'recency-focused', or 'ICL'
    ) -> str:
    
    def get_item_metadata(item_id: str) -> Dict[str, str]:
        """Retrieve metadata for the given item ID, or default if not found."""
        return item_features.get(item_id, DEFAULT_ITEM_METADATA)

    def format_interaction(idx: int, item_id: int, args: Dict[str, bool]) -> str:
        """
        Format a single interaction into a detailed string with optional summary, rating, and review.
        """
        item_title = get_item_metadata(item_id).get("title", "Unknown")
        interaction_text = f"{idx + 1}. `{item_title}`"
        return interaction_text

    # Generate formatted user history
    def generate_user_history_text(target_user_history: List[Dict[str, str]], args: Dict[str, bool], add_prompt=True) -> str:
        """
        Generate formatted text for the user's purchase history based on their interactions.
        """
        # prompt = f"Here are the products {args.target} purchased by the target user in the past:\n"
        prompt = f"Here are the purchase history of the target user:\n"
        instruct = f"\nThink step by step and select the best product from the candidate list that the user is most likely to purchase next."
        interaction_texts = [format_interaction(idx, interaction, args) for idx, interaction in enumerate(target_user_history)]
        if not add_prompt:
            return "\n".join(interaction_texts)
        if args.prompt_type == "llama_user_pref":
            return prompt + "\n".join(interaction_texts)
        last_purchase_prompt = f"\nThe user's most recent purchase: {interaction_texts[-1]}"
        return prompt + "\n".join(interaction_texts)+last_purchase_prompt  + instruct
    
    # Generate user history text
    target_user_history = target_user_history[-args.max_history_len:]
    user_his_text = generate_user_history_text(target_user_history, args)
    user_his_text_no_recent = generate_user_history_text(target_user_history[:-1], args)

    # Get the most recent item for the prompt
    recent_item_id = int(target_user_history[-1])
    recent_item = get_item_metadata(recent_item_id).get("title", "Unknown")

    # Construct the prompt using the unified format from construct_prompt function
    if prompt_type == "sequential":
        prompt =f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                f"Now there are {args.num_candidates} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                f"Please rank these {args.num_candidates} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                f"Please show me your ranking top-10 results with order numbers. Split your output with line break. You MUST rank the given candidate products. You can not generate products that are not in the given candidate list." \

    elif prompt_type == "recency-focused":
        prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                f"Now there are {args.num_candidates} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                f"Please rank these {args.num_candidates} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                f"Note that my most recently purchased product is `{recent_item}`. " \
                f"Please show me your ranking top-10 results with order numbers. Split your output with line break. You MUST rank the given candidate products. You can not generate products that are not in the given candidate list."
    
    elif prompt_type == "ICL":
        prompt = f"I've purchased the following products in the past in order:\n{user_his_text_no_recent}\n\n" \
                f"Then if I ask you to recommend a new product to me according to the given purchasing history, you should recommend `{recent_item}` and " \
                f"now that I've just purchased `{recent_item}`, " \
                f"there are {args.num_candidates} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                f"Please rank these {args.num_candidates} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                f"Please show me your ranking top-10 results with order numbers. Split your output with line break. You MUST rank the given candidate products. You can not generate products that are not in the given candidate list."
    
    if prompt_type in ["sequential", "recency-focused", "ICL", "seq_cand"]:
        # prompt += "Don't add any additional information or comments in the response."
        prompt += "\nThe response must contain the ranked product names only, starting from the most likely to the least likely, with each product on a new line. Do not include any additional comments."

    return prompt, candidate_text_order

def get_reference_text(args, similar_user_history_list, item_features):
    reference_text = ""
    for i, user_history_data in enumerate(similar_user_history_list):
        # reference_text += f"##### Reference User {i+1}: "
        history_descriptions = []
        for j, interaction in enumerate(user_history_data["sequence"][-args.ref_user_seq_len:]):
            item_id = interaction["item_id"]
            # time = interaction["time"]
            item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
            item_text = f"{j+1}. `{item_metadata[args.target]}`"
            # item_text = f"`{item_metadata[args.target]}`"
            # if args.use_time:
                # item_text += f"({time} days ago)"
            if args.use_summary:
                item_text += f": {item_metadata['summary'].strip()}"
            history_descriptions.append(item_text)
        histor_text = "\n".join(history_descriptions)
        # history_text = ", ".join(history_descriptions[:-1]) + (
        #     f", and {history_descriptions[-1]}" if len(history_descriptions) > 1 else history_descriptions[0]
        # )
        reference_text += f"\n#### Here are the history of reference user {i+1}:\n{histor_text}.\n"

        label_item_id = user_history_data["target_item_id"]
        rationale = user_history_data["rationale"]
        label_item_metadata = item_features.get(label_item_id, DEFAULT_ITEM_METADATA)
        label_item_summary = label_item_metadata.get('summary', '')
        candidate_text = get_cand_list_text(user_history_data["candidate_item_ids"], item_features, args, spliter="\n")
        reference_text += (
            # f"Here are candidate items:\n{candidate_text}. "
            f"This user will purchase `{label_item_metadata['title']}`. {rationale}\n"
            # f"This user will purchase `{label_item_metadata['title']}`, {label_item_summary}\n"
        )

    instruction = "Now, use the patterns from these examples to predict the next purchase for the following target user history.\n"
    return reference_text + instruction



def generate_llm_reranking_prompt(
        args: Dict[str, object], 
        target_user_history: List[int], 
        item_features: Dict[int, Dict[str, str]], 
        candidate_text_order: str,  # Candidates list in the form of a string
    ) -> str:
    
    def get_item_metadata(item_id: int) -> Dict[str, str]:
        """Retrieve metadata for the given item ID, or return default values if not found."""
        return item_features.get(item_id, {"title": "Unknown", "brand": "Unknown", "categories": "Unknown", "description": "Unknown"})

    def format_interaction(idx: int, item_id: int) -> str:
        """
        Format a single interaction into a detailed string for the purchase history.
        """
        item_title = get_item_metadata(item_id).get("title", "Unknown")
        brand = get_item_metadata(item_id).get("brand", "Unknown")
        categories = get_item_metadata(item_id).get("categories", "Unknown")
        # flatten the categories 2d list
        categories = ", ".join([cat for sublist in categories for cat in sublist])
        return f"{idx + 1}. `{item_title}`\n- Brand: {brand}\n- Categories: {categories}"

    # Generate formatted user purchase history
    def generate_user_history_text(target_user_history: List[int], add_prompt=True) -> str:
        """
        Generate formatted text for the user's purchase history.
        """
        prompt = "Here is the purchase history of the target user:\n"
        interaction_texts = [format_interaction(idx, item_id) for idx, item_id in enumerate(target_user_history)]
        
        if not add_prompt:
            return "\n".join(interaction_texts)
        
        last_purchase_prompt = f"\nThe user's most recent purchase: `{interaction_texts[-1]}`"
        instruct = "\nNow, analyze this purchase pattern and rank the candidate products based on their likelihood of being the user's next purchase."
        return prompt + "\n".join(interaction_texts) + last_purchase_prompt + instruct
    
    # Generate user history text
    target_user_history = target_user_history[-args.max_history_len:]
    user_his_text = generate_user_history_text(target_user_history)

    # Construct the ranking prompt
    prompt = f"""
You are an intelligent assistant designed to predict the next item a user is likely to purchase based on their past purchasing history.
Your task is to analyze the provided user history and rank the following candidate items in order of purchase likelihood.

### **User Purchase History:**
{user_his_text}

### **Candidate Items:**
{candidate_text_order}

### **Task:**
Think step by step and rank the candidate items based on their likelihood of being the user's next purchase.

### **Output Format:**
The response must contain the ranked product titles only, starting from the most likely to the least likely.
Each product title must be on a separate line with **no additional text**.
""".strip()

    return prompt, candidate_text_order

def get_reference_text(args, similar_user_history_list, item_features):
    reference_text = ""
    for i, user_history_data in enumerate(similar_user_history_list):
        # reference_text += f"##### Reference User {i+1}: "
        history_descriptions = []
        for j, interaction in enumerate(user_history_data["sequence"][-args.ref_user_seq_len:]):
            item_id = interaction["item_id"]
            # time = interaction["time"]
            item_metadata = item_features.get(item_id, DEFAULT_ITEM_METADATA)
            item_text = f"{j+1}. `{item_metadata[args.target]}`"
            # item_text = f"`{item_metadata[args.target]}`"
            # if args.use_time:
                # item_text += f"({time} days ago)"
            if args.use_summary:
                item_text += f": {item_metadata['summary'].strip()}"
            history_descriptions.append(item_text)
        histor_text = "\n".join(history_descriptions)
        # history_text = ", ".join(history_descriptions[:-1]) + (
        #     f", and {history_descriptions[-1]}" if len(history_descriptions) > 1 else history_descriptions[0]
        # )
        reference_text += f"\n#### Here are the history of reference user {i+1}:\n{histor_text}.\n"

        label_item_id = user_history_data["target_item_id"]
        rationale = user_history_data["rationale"]
        label_item_metadata = item_features.get(label_item_id, DEFAULT_ITEM_METADATA)
        label_item_summary = label_item_metadata.get('summary', '')
        candidate_text = get_cand_list_text(user_history_data["candidate_item_ids"], item_features, args, spliter="\n")
        reference_text += (
            # f"Here are candidate items:\n{candidate_text}. "
            f"This user will purchase `{label_item_metadata['title']}`. {rationale}\n"
            # f"This user will purchase `{label_item_metadata['title']}`, {label_item_summary}\n"
        )

    instruction = "Now, use the patterns from these examples to predict the next purchase for the following target user history.\n"
    return reference_text + instruction

SYSTEM_PROMPT = """
You are an intelligent recommendation assistant specializing in predicting the most relevant next item a user is likely to purchase.
Your task is to analyze the user's preferences based on their purchase history and reviews, and generate a well-reasoned recommendation for their next potential purchase.
"""

NEXT_ITEM_PROMPT = """
### Task:
Predict **next item title** that the user is likely to purchase based on the user's purchase history and preferences.

### User Purchase History:
{user_history}

 ### Response:
"""

NEXT_BRAND_PROMPT = """
### Task:
Predict **next item brand** that the user is likely to purchase based on the user's purchase history and preferences.

### User Purchase History:
{user_history}

 ### Response:
"""

NEXT_CATEGORY_PROMPT = """
### Task:
Predict **next item categories** that the user is likely to purchase based on the user's purchase history and preferences.

### User Purchase History:
{user_history}

 ### Response:
"""

NEXT_DESCRIPTION_PROMPT = """
### Task:
Predict **next item description** that the user is likely to purchase based on the user's purchase history and preferences.

### User Purchase History:
{user_history}

 ### Response:
"""

NEXT_FEATURES_PROMPT = """
### Task:
Predict **next item features** that the user is likely to purchase based on the user's purchase history and preferences.

### User Purchase History:
{user_history}

 ### Response:
"""


def get_prompt(prompt_type):
    if prompt_type == "next_item_title":
        return NEXT_ITEM_PROMPT
    elif prompt_type == "next_item_brand":
        return NEXT_BRAND_PROMPT
    elif prompt_type == "next_item_category":
        return NEXT_CATEGORY_PROMPT
    elif prompt_type == "next_item_description":
        return NEXT_DESCRIPTION_PROMPT
    elif prompt_type == "next_item_features":
        return NEXT_FEATURES_PROMPT
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")


# RESPONSE_TEMPLATE = " ### Response:"

def get_response_template(model_name):
    model_name = model_name.lower()
    if "gemma" in model_name:
        return "<start_of_turn>model"
    elif "llama" in model_name:
        return "<|start_header_id|>assistant<|end_header_id|>"
    else:
        return None
        # raise ValueError()
RESPONSE_TEMPLATE = "<start_of_turn>model"
