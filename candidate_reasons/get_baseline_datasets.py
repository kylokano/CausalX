# Get the baseline datasets to make sure the comperation of model is fair
import json
import os
# with open("candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_CDs_and_Vinyl_small.json", 'r', encoding='utf8') as f:
#     raw_data = json.load(f)
# writter = open(
#     "baselines/baseline_datasets/Amazon_CDs_and_Vinyl_small_b/Amazon_CDs_and_Vinyl_small_b.inter", 'w', encoding='utf8')
# writter.write("user_id:token\titem_id:token\n")
# for user_id, value in raw_data.items():
#     for pos_item_id in value["pos_item"]:
#         writter.write(str(user_id)+"\t"+str(pos_item_id)+"\n")

# writter.close()

# with open("candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_ml-latest-small.json", 'r', encoding='utf8') as f:
#     raw_data = json.load(f)
# writter = open(
#     "baselines/baseline_datasets/ml-latest-small_b/ml-latest-small_b.inter", 'w', encoding='utf8')
# writter.write("user_id:token\titem_id:token\n")
# for user_id, value in raw_data.items():
#     for pos_item_id in value["pos_item"]:
#         writter.write(str(user_id)+"\t"+str(pos_item_id)+"\n")

with open("candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_Books_small.json", 'r', encoding='utf8') as f:
    raw_data = json.load(f)
os.makedirs("baselines/baseline_datasets/Amazon_Books_small_b", exist_ok=True)
writter = open(
    "baselines/baseline_datasets/Amazon_Books_small_b/Amazon_Books_small_b.inter", 'w', encoding='utf8')
writter.write("user_id:token\titem_id:token\n")
for user_id, value in raw_data.items():
    for pos_item_id in value["pos_item"]:
        writter.write(str(user_id)+"\t"+str(pos_item_id)+"\n")
