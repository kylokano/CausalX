#!/bin/bash
learning_rate=0.001
ui_ratio=0
reason_ratio=1

neg_user_loss=true
ur_i_loss=true
ir_u_loss=true
u_ir_loss=false
reg_loss=true

pop_bucket=10
debias_coeffiecient=1
bucket_overlap=3
pop_ratio=0.01

python CF_model_train_debias.py \
    --dataset=cluster_results_MiniLM_UMAP20_openai_amazoncd \
    --gpu_id=3 \
    --learning_rate=$learning_rate \
    --is_pairwise=False \
    --ui_ratio=$ui_ratio \
    --reason_ratio=$reason_ratio \
    --neg_user_loss=$neg_user_loss \
    --ur_i_loss=$ur_i_loss \
    --ir_u_loss=$ir_u_loss \
    --u_ir_loss=$u_ir_loss \
    --reg_loss=$reg_loss \
    --pop_bucket=$pop_bucket \
    --debias_coeffiecient=$debias_coeffiecient \
    --bucket_overlap=$bucket_overlap \
    --pop_ratio=$pop_ratio \
    --wandb_project=new_reason_CF