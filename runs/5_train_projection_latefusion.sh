#!/bin/bash



# models=(gemma-3-12b-it)
# models=(Qwen3-Next-80B-A3B-Instruct)
models=(gemma-3-1b-it)
# models=(gemma-3-12b-it)

datasets=(toys beauty)
datasets=(beauty)
datasets=( yelp )
datasets=(toys)
datasets=(beauty sports toys)
emb_models=(
  mixedbread-ai/mxbai-embed-large-v1
  # mixedbread-ai/mxbai-embed-xsmall-v1
  # jinaai/jina-embeddings-v2-base-en
  # Qwen/Qwen3-Embedding-0.6B
  # google/embeddinggemma-300m 
  # Alibaba-NLP/gte-base-en-v1.5
  # BAAI/bge-large-en-v1.5
)

run_name="encoder_target_pref_reasoning"
item_feature_name="item_preference"
target_name="target_preference_reasoning"


for model_name in "${models[@]}"; do
for dataset in "${datasets[@]}"; do
# for seed in 22 32 42 42 52 ; do
for embedding_model_name in "${emb_models[@]}"; do
# for beta in 0.2 0.05 0.005; do
for seed in 42 ; do
for beta in 0.5 ; do
  echo "runs train projection $dataset..."
  CUDA_VISIBLE_DEVICES=1 python3 src/train_alignment.py \
  --run_name $run_name \
  --data_name $dataset \
  --item_target_names_str $item_feature_name \
  --embedding_model_name $embedding_model_name \
  --num_neg_samples 99 \
  --train_batch_size 128 \
  --eval_batch_size 512 \
  --embedding_dim 512 \
  --learning_rate 1e-4 \
  --late_fusion \
  --seed $seed \
  --model_name $model_name \
  --epochs 20 \
  --alpha $alpha 
done
done
done
done
done
  --wo_recsys \
  --zero_shot \
  --use_pref \
  # --item_target_names_str review_description \
  # --use_pre_train \
  # --zero_shot \
  # --filter_top_k 500
  # --days 14 \
  # --beta $beta \
  # --embedding_model_name sentence-t5-large \
  # --token_limit 64 \
  # --item_target_names_str review_description \

# for model_name in "${models[@]}"; do
# for dataset in "${datasets[@]}"; do
#   echo "runs train emb encoder $dataset..."
#   CUDA_VISIBLE_DEVICES=2 python3 src/train_embedding.py \
#   --run_name pref \
#   --data_name $dataset \
#   --item_target_names_str review_description \
#   --epochs 5 \
#   --num_neg_samples 99 \
#   --train_batch_size 128 \
#   --days 14 \
#   --embedding_dim 1024 \
#   --learning_rate 1e-4 \
#   --model_name $model_name
# done
# done
  # --embedding_model_name "meta-llama/Llama-3.2-3B-Instruct" \

  # --add_item_meta \
# for model_name in "${models[@]}"; do
# for dataset in "${datasets[@]}"; do
#   echo "runs train emb encoder $dataset..."
#   CUDA_VISIBLE_DEVICES=1 python3 src/train_embedding.py \
#   --run_name pref_only \
#   --data_name $dataset \
#   --item_target_names_str review_description \
#   --epochs 20 \
#   --num_neg_samples 99 \
#   --train_batch_size 64 \
#   --days 14 \
#   --use_item_meta \
#   --learning_rate 1e-4 \
#   --model_name $model_name
# done
# done