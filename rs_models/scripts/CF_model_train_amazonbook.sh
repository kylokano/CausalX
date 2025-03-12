#!/bin/bash
learning_rates=(0.005 0.001 0.0005)
# learning_rates=(0.005)

# start_reasons_ratio=(0.01 0.01 0.1 0.01)
# end_reasons_ratio=(100 100 10 100)
# # start_epoch=(10 10 10 5 5)
# # end_epoch=(40 30 20 15 25)
# start_epoch=(10 0 0 0)
# end_epoch=(30 10 10 1)

# ui_ratios=(0)
ui_ratios=(0 0.001 0.01 0.05)
reason_ratio=1

neg_user_loss=(true false)
ur_i_loss=(false)
ir_u_loss=(false)
u_ir_loss=(false)
reg_loss=(true)

# Loop through all arrays by index
for lr in "${learning_rates[@]}"; do
    for ui_r in "${ui_ratios[@]}"; do
        for neg in "${neg_user_loss[@]}"; do
            for ur in "${ur_i_loss[@]}"; do
                for ir in "${ir_u_loss[@]}"; do
                    for uir in "${u_ir_loss[@]}"; do
                        for rgl in "${reg_loss[@]}"; do
                        # Command to run the Python script
                            python CF_model_train.py --dataset=cluster_results_MiniLM_UMAP20_openai_amazonbooks --gpu_id=1 --learning_rate=$lr --is_pairwise=False --ui_ratio=$ui_r --reason_ratio=$reason_ratio --neg_user_loss=$neg --ur_i_loss=$ur --ir_u_loss=$ir --u_ir_loss=$uir --reg_loss=$rgl --wandb_project=new_reason_CF
                            # KeyboardInterrupt
                            break
                        done
                    done
                done
            done
        done
    done
done
#     done
#     for ((i=0; i<$array_length; i++)); do
#         startr=${start_reasons_ratio[$i]}
#         endr=${end_reasons_ratio[$i]}
#         s=${start_epoch[$i]}
#         e=${end_epoch[$i]}
        
#         # Command to run the Python script
#         python CF_model_train.py --gpu_id=4 --learning_rate=$lr --start_reasons_ratio=$startr --end_reasons_ratio=$endr --start_epoch=$s --end_epoch=$e --is_pairwise=False
#     done
# done