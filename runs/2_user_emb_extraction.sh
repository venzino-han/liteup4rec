#!/bin/bash

datasets=(beauty sports yelp toys)
datasets=(yelp)
emb_models=(
    mixedbread-ai/mxbai-embed-large-v1
    # mixedbread-ai/mxbai-embed-xsmall-v1
    # Qwen/Qwen3-Embedding-0.6B
    # BAAI/bge-large-en-v1.5
  )


model_names=(
  gemma-3-1b-it
  # gemma-3-4b-it
)



item_feature_name="item_preference"
target_names=(user_preference)
run_name="user_preference"

for model_name in "${model_names[@]}"; do
for dataset in "${datasets[@]}"; do
for target_name in $target_names; do
for embedding_model_name in "${emb_models[@]}"; do
  echo "Processing $dataset..."
  CUDA_VISIBLE_DEVICES=0 python3 src/extract_user_emb.py \
  --run_name $run_name \
  --data_name "$dataset" \
  --device "cuda:0" \
  --batch_size 512 \
  --model_name $model_name \
  --embedding_model_name $embedding_model_name \
  --use_sentence_transformers \
  --target $target_name 
done
done
done
done
  # --add_item_meta \
