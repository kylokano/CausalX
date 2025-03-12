#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# 超参数选项
item_features=(only_name base_features)
prompt_type=(zero_shot)
reason_numbers=(3 5 10)

for if in "${item_features[@]}"; do
    for pt in "${prompt_type[@]}"; do
        echo "Training model with item_features: $if, prompt_type: $pt, reason_number: $rn"
        python LLMRS_prompt.py --item_features=$if --prompt_type=$pt --gpu_id=1 --exp_name=llama2-13b-amazoncd-$pt-$if
    done
done

