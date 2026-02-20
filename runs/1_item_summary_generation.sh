#!/bin/bash

export VLLM_USE_V1=1

datasets=(toys beauty sports)
datasets=(yelp)
model_name="google/gemma-3-4b-it"
for dataset in "${datasets[@]}"
do
  echo "Query generation $dataset..."
  CUDA_VISIBLE_DEVICES=0 python3 src/generate_item_summary.py \
  --run_name "item_preference" \
  --max_words 512 \
  --max_output_tokens 1024 \
  --data_name $dataset \
  --model_name $model_name
done
