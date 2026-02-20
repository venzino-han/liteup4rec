#!/bin/bash

# List of dataset names
datasets=(beauty toys sports)
datasets=(yelp)

# List of models to use
models=(gemma-3-1b-it)

# List of embedding models
emb_models=(mixedbread-ai/mxbai-embed-large-v1)

# Training configuration
# run_name="encoder_zeroshot"

run_name="encoder_target_pref_reasoning"
item_feature_name="item_preference"
target_name="target_preference_reasoning"

# Loop through all combinations
for model_name in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for embedding_model_name in "${emb_models[@]}"; do
        echo "Training encoder for dataset: $dataset, model: $model_name, embedding: $embedding_model_name"
        CUDA_VISIBLE_DEVICES=0 python3 src/train_user_encoder.py \
          --run_name $run_name \
          --data_name $dataset \
          --device cuda:0 \
          --seed 42 \
          --num_heads 2 \
          --num_layers 1 \
          --max_seq_length 8 \
          --train_batch_size 128 \
          --eval_batch_size 512 \
          --learning_rate 1e-3 \
          --epochs 30 \
          --early_stop 5 \
          --mse_weight 1.0 \
          --infonce_weight 0.0 \
          --cosine_weight 0.0 \
          --item_target_names_str $item_feature_name \
          --model_name $model_name \
          --embedding_model_name $embedding_model_name \
          --token_limit 1024
    done
  done
done
          # --reject_sampling \
        #   --use_special_tokens \
