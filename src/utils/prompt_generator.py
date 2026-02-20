import argparse
import random
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset

from utils.prompt import (
    get_prompt,
    SYSTEM_PROMPT,
)

def generate_prompt_from_user_sequence(item_ids, item_metadata,  args):
    """
    Generates a formatted prompt from a single user sequence.
    
    Args:
        item_ids (list): List of item IDs in the user's sequence.
        item_features (dict): Dictionary mapping item IDs to item features.
    Returns:
        str: Formatted prompt ready for LLM input.
    """

    # List to store formatted history
    history_text_list = []

    # Process each item in the user's history
    for item_id in item_ids:
        item_data = item_metadata.get(item_id)
        if item_data is None:
            raise ValueError(f"Item ID {item_id} not found in item features.")

        item_title = item_data.get('title')
        item_brand = item_data.get('brand')
        item_categories = item_data.get('category')
        item_description = item_data.get('description')
        item_features = item_data.get('features')

        # Base history format
        item_history_text = f"**Title:** `{item_title}`"
        
        if args.use_brand:
            item_history_text += f"\n**Brand:** {item_brand}"
        if args.use_category:
            item_history_text += f"\n**Categories:** {item_categories}"
        if args.use_description:
            item_description = item_description.replace("\n", " ")
            if len(item_description.split()) > args.history_text_max_length:
                item_description = " ".join(item_description.split()[:args.history_text_max_length]) + "..."
                item_history_text += f"\n**Description:** {item_description}"
        
        if args.use_review:
            # TODO
            pass

        if args.use_features:
            if len(item_features.split("\n")) > 10:
                item_features = "\n".join(item_features.split("\n")[:10])
            item_features = item_features.replace("\n-", ",").replace("- ", "")
            item_history_text += f"\n**Features:**\n{item_features}"

        history_text_list.append(item_history_text)

    # Apply history length constraint
    if len(history_text_list) > args.max_history_len:
        history_text_list = history_text_list[-args.max_history_len:]

    # Construct the final history text
    history_text = "\n---\n".join(f"{i+1}. {history}" for i, history in enumerate(history_text_list))

    # Add final note about the most recent item
    if args.use_last_item:
        last_item_title = item_metadata.get(item_ids[-1], {}).get('title')
        history_text += f"\n\n`{last_item_title}` is the most recently purchased item."

    return history_text

def apply_chat_template(system_prompt, prompt, response, tokenizer):
    # messages = [
    #     [
    #         {
    #             "role": "system",
    #             "content": [{"type": "text", "text": system_prompt},]
    #         },
    #         {
    #             "role": "user",
    #             "content": [{"type": "text", "text": prompt},]
    #         },
    #         {
    #             "role": "assistant",
    #             "content": [{"type": "text", "text": response},]
    #         },
    #     ],
    # ]
    messages = [
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": response
            },
        ],
    ]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    return formatted[0]

class TrainPromptGenerator:
    def __init__(self, args: argparse.Namespace, item_metadata: Dict[int, Dict[str, Any]], uid_to_seq:Dict[int,List[int]], tokenizer):
        """
        Initializes the prompt generator.

        Args:
            args (argparse.Namespace): Training arguments.
            item_features (dict): Dictionary mapping item IDs to item features.
            uid_to_seq (dict): Dictionary mapping user IDs to user sequences.
        """
        self.args = args
        self.item_metadata = item_metadata
        self.uid_to_seq = uid_to_seq
        self.tokenizer = tokenizer


    def generate_sft_dataset(self, response_template, split="train",) -> Dataset:
        """
        Generates a dataset for Supervised Fine-Tuning (SFT).

        Args:
            sft_data_list (list): List of training data.

        Returns:
            Dataset: Hugging Face dataset.
        """
        if split == "train":
            idx = -3
            max_history_len = self.args.train_max_history_len
        elif split == "valid":
            idx = -2
            max_history_len = self.args.max_history_len
        elif split == "test":
            idx = -1
            max_history_len = self.args.max_history_len

        prompt_list = []
        label_text_list = []
        for uid, seq in self.uid_to_seq.items():
            target_item_id = seq[idx]
            seq = seq[:idx][-max_history_len:]
            user_history_text = generate_prompt_from_user_sequence(seq, self.item_metadata, self.args)

            # apply chat template
            target_names = self.args.target_names
            for target_name in target_names:
                target_text = self.item_metadata[target_item_id][target_name]
                if target_name == "features":
                    target_text = target_text.replace("\n-", ",").replace("- ", "")
                prompt_format = get_prompt(f"next_item_{target_name}")
                prompt = prompt_format.format(
                        user_history=user_history_text
                    )
                formatted = apply_chat_template(SYSTEM_PROMPT, prompt, target_text, self.tokenizer)
                prompt, label = formatted.split(response_template)
                prompt = prompt + response_template
                # label = label + self.tokenizer.eos_token

                prompt_list.append(prompt)
                label_text_list.append(label)

            if uid == 10:
                print(f"uid: {uid}\n")
                print(f"history:\n{user_history_text[-500:]}\n")
                print(f"Target:\n{target_text}")
                print(prompt)
                print(label)
                print(f"-"*50)

        # Create DataFrame and shuffle

        if split == "valid":
            assert len(prompt_list) == len(label_text_list)
            combined = list(zip(prompt_list, label_text_list))
            sampled = random.sample(combined, len(combined) // 20)
            prompt_list, label_text_list = zip(*sampled)

        if split == "train":
            # drop label text "Unknown" case
            print(f"train samples before filtering: {len(prompt_list)}")
            valid_indices = [i for i, label in enumerate(label_text_list) if "Unknown" not in label]
            prompt_list = [prompt_list[i] for i in valid_indices]
            label_text_list = [label_text_list[i] for i in valid_indices]
            assert len(prompt_list) == len(label_text_list)
            print(f"train samples after filtering: {len(prompt_list)}")

        if split == "train" and len(prompt_list) > self.args.train_samples:
            indices = random.sample(range(len(prompt_list)), self.args.train_samples)
            uid_list = [uid for uid, seq in self.uid_to_seq.items() for _ in range(len(seq))]
            prompt_list = [prompt_list[i] for i in indices]
            label_text_list = [label_text_list[i] for i in indices]
        elif split in ["valid", "test"] and len(prompt_list) > self.args.test_samples:
            prompt_list = prompt_list[:self.args.test_samples] 
            label_text_list = label_text_list[:self.args.test_samples]
            
        dataset = Dataset.from_dict({
            # "uids":uid_list, 
            "prompt": prompt_list, "label": label_text_list})
        # data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return dataset

    def generate_dpo_dataset(self, sft_data_list: List[Dict]) -> Dataset:
        """
        Generates a dataset for Direct Preference Optimization (DPO).

        Args:
            sft_data_list (list): List of training data.

        Returns:
            Dataset: Hugging Face dataset.
        """
        prompt_list = []
        chosen_list = []
        rejected_list = []

        for data in sft_data_list:
            target_query = self.item_features[data["target_item_id"]][self.args.sft_target]
            candidate_item_ids = data["candidate_item_ids"]
            candidate_item_ids.remove(data["target_item_id"])

            # history_seq = data["sequence"][-self.args.max_history_len:]
            # user_history_text = generate_target_user_history(
            #     self.args, history_seq, self.item_features
            # )

            # # Format prompt
            # prompt = generate_llama_instruction_prompt(
            #     instruction=QUERY_GENERATION_INSTRUCTION.strip(),
            #     input_text=user_history_text
            # ).strip()

            # chosen = generate_llama_response_prompt(target_query)

            # # Process rejected queries
            # for rejected_query in data["rejected_query"]:
            #     rejected = generate_llama_response_prompt(rejected_query)
            #     prompt_list.append(prompt)
            #     chosen_list.append(chosen)
            #     rejected_list.append(rejected)

        # Create DataFrame and shuffle
        data_df = pd.DataFrame({"prompt": prompt_list, "chosen": chosen_list, "rejected": rejected_list})

        if len(data_df) > self.args.num_pairs_for_train:
            data_df = data_df.sample(n=self.args.num_pairs_for_train, random_state=42).reset_index(drop=True)

        return Dataset.from_pandas(data_df)