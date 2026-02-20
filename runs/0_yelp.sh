# run all process for yelp dataset
# !/bin/bash
export VLLM_USE_V1=1

# ./runs/1_reasoning_generation.sh
./runs/1_item_summary_generation.sh
./runs/2_item_emb_extraction.sh
./runs/2_user_emb_extraction.sh
./runs/3_train_profile_surrogates.sh
./runs/4_train_projection.sh







