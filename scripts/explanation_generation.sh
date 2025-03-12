export CUDA_VISIBLE_DEVICES=3
datasets=("ml-latest-small" "Amazon_CDs_and_Vinyl_small" "Amazon_Books_small")

for dataset in ${datasets[@]}; do
    # get the prompts for explanations generation
    python get_rs_reasons_prompt.py --dataset $dataset

    # generate explanations with API
    python candidate_reasons/openai.py --dataset $dataset
    python candidate_reasons/correct_id.py --dataset $dataset

    # construct the database of candidate explanations
    python candidate_reasons/dbscan_embedding_openai.py --dataset $dataset
done

