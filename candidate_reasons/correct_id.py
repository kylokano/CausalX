import re
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-latest-small")
    args = parser.parse_args()
    dataset_name = args.dataset


    if dataset_name == "Amazon_CDs_and_Vinyl_small":
        raw_get_reasons_file_name = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_CDs_and_Vinyl_small.json"
        # corrected_raw_get_reasons_file_name = "get_openai_reasons_amazoncd.json"
        raw_openai_reasons_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_amazoncd.txt"
        corrected_openai_reasons_file_name = "candidate_reasons/raw_generated_reasons/openai_reasons_amazoncd.json"

    elif dataset_name == "ml-latest-small":
        raw_get_reasons_file_name = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_ml-latest-small.json"
        # corrected_raw_get_reasons_file_name = "get_openai_reasons_movielens.json"
        raw_openai_reasons_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_ml100k.txt"
        corrected_openai_reasons_file_name = "candidate_reasons/raw_generated_reasons/openai_reasons_movielens.json"
    elif dataset_name == "Amazon_Books_small":
        raw_get_reasons_file_name = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_Books_small.json"
        raw_openai_reasons_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_amazonbooks.txt"
        corrected_openai_reasons_file_name = "candidate_reasons/raw_generated_reasons/openai_reasons_amazonbooks.json"

    reasons_output = []
    all_reasons = []
    with open(raw_openai_reasons_file_name, 'r', encoding='utf8') as f:
        for line in f:
            openai_reasons = json.loads(line)
            if dataset_name == "Amazon_Books_small":
                model_user_id, model_item_id = openai_reasons["union_id"].split(
                    "-")
            else:
                model_user_id, model_item_id = openai_reasons["union_id"].split(
                    ";")

            raw_user_id, raw_item_id = model_user_id, model_item_id
            pattern = re.compile(r'^\s*-\s+|^\s*\d+\.\s+')
            reasons_list = [pattern.sub('', i)
                            for i in openai_reasons['reasons'].split("\n")]
            reasons_output.append(
                {"user_id": raw_user_id, "item_id": raw_item_id, "reasons": reasons_list})
            for tmp in reasons_list:
                all_reasons.append(tmp)
    with open(corrected_openai_reasons_file_name, 'w', encoding='utf8') as f:
        json.dump(reasons_output, f)