import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingExtractor:
    def __init__(self, args):
        """
        Initialize the embedding extractor.

        Args:
            model_name (str): Name of the SentenceTransformer model.
            device (str): Device to use for processing (e.g., 'cuda:0', 'cpu').
            batch_size (int): Batch size for embedding extraction.
        """
        self.model_name = args.embedding_model_name
        self.device = args.device
        self.batch_size = args.batch_size
        
        # Load embedding model
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"torch_dtype": torch.float32},
        )

    def extract_embeddings(self, input_text_list: List[str], batch_size: int = None) -> torch.Tensor:
        """
        Extract embeddings for a list of input texts.

        Args:
            input_text_list (List[str]): List of input texts to process.
            batch_size (int, optional): Batch size for processing embeddings. Defaults to self.batch_size.

        Returns:
            torch.Tensor: A tensor containing the embeddings for the input texts.
        """
        batch_size = batch_size or self.batch_size
        embeddings = []

        # Process embeddings in batches
        for i in range(0, len(input_text_list), batch_size):
            batch_texts = input_text_list[i:i + batch_size]
            if "gemma" in self.model_name:
                instruction = 'Please provide a brief description of the item that simulates the user query.'
                prompt = f'<instruct>{instruction}\n<query>'
                batch_embeddings = self.model.encode(batch_texts, prompt=prompt, convert_to_tensor=True, show_progress_bar=False).cpu()
            else:
                batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False).cpu()
            embeddings.append(batch_embeddings)

        # Combine all embeddings into a single tensor
        embeddings_tensor = torch.cat(embeddings, dim=0)
        return embeddings_tensor


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings for a list of input texts.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the embedding model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for embedding extraction.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction.")
    args = parser.parse_args()

    # Example input text list
    input_text_list = [
        "A compact and efficient coffee maker with adjustable settings for a personalized experience.",
        "A durable and stylish backpack, perfect for outdoor adventures and daily use.",
        "A lightweight and ergonomic wireless mouse with customizable buttons for enhanced productivity."
    ]

    extractor = EmbeddingExtractor(
        model_name=args.embedding_model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    embeddings = extractor.extract_embeddings(input_text_list)

    # Save embeddings to a file
    output_file = "output_embeddings.pt"
    torch.save(embeddings, output_file)
    print(f"Embeddings saved to {output_file}")

