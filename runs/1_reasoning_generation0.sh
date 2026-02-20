#!/bin/bash

# export VLLM_USE_V1=1
export VLLM_USE_V1=1

datasets=(toys beauty sports)
splits=(train valid test)
models=(
  google/gemma-3-1b-it 
  google/gemma-3-4b-it 
  )

for model_name in "${models[@]}"; do
for dataset in "${datasets[@]}"; do
for split in "${splits[@]}"; do
  echo "Query generation $dataset..."
  CUDA_VISIBLE_DEVICES=0 python3 src/generate_user_preference.py \
  --run_name "user_preference" \
  --data_name $dataset \
  --split $split \
  --model_name $model_name
done
done
done
