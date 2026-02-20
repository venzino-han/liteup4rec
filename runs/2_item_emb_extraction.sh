#!/bin/bash

datasets=(toys sports beauty)
datasets=(yelp)

model_name="gemma-3-4b-it"
emb_models=(
  mixedbread-ai/mxbai-embed-large-v1
  # mixedbread-ai/mxbai-embed-xsmall-v1
  # Qwen/Qwen3-Embedding-0.6B
  # BAAI/bge-large-en-v1.5
)
for dataset in "${datasets[@]}"; do
# for target_name in "item_feature_no_review"; do
# for target_name in "review_description"; do
for target_name in "item_preference"; do
for embedding_model_name in "${emb_models[@]}"; do
  echo "Processing $dataset..."
  CUDA_VISIBLE_DEVICES=1 python3 src/extract_item_emb.py \
  --data_name "$dataset" \
  --device "cuda:0" \
  --add_item_meta \
  --batch_size 512 \
  --model_name $model_name \
  --embedding_model_name $embedding_model_name \
  --use_sentence_transformers \
  --target $target_name 
done
done
done
