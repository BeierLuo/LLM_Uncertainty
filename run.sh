#!/bin/bash

# 设置默认值
model=llama2_chat_7B #opt-6.7b 
dataset_name=tqa
cuda=$1

echo '=============Get LLM generations==========='
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dataset_name --model_name $model --most_likely 1 --num_gene 1 --gene 1 --cuda $cuda

echo '=============Get the fround truth for the LLM generations============'
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dataset_name --model_name $model --most_likely 1 --use_rouge 0 --generate_gt 1 --cuda $cuda

echo '=============Hallucination Detection==============='
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dataset_name --model_name $model --use_rouge 0 --most_likely 1 --weighted_svd 1 --feat_loc_svd 3 --cuda $cuda
