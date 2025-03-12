from tqdm import tqdm
import requests
import json
import time
import argparse


def get_response(prompt):
    # Set up the payload for chat-based models
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "n": 1,
        "temperature": 0,
    }

    # Send the POST request
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    # Extract and print the response
    try:
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']
    except:
        assistant_message = "Error"
    return assistant_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-latest-small")
    args = parser.parse_args()
    datasets_name = args.dataset

    if datasets_name == "ml-latest-small":
        file_path = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_ml-latest-small.json"
        out_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_ml100k.txt"
    elif datasets_name == "Amazon_CDs_and_Vinyl_small":
        file_path = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_CDs_and_Vinyl_small.json"
        out_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_amazoncd.txt"
    elif datasets_name == "Amazon_Books_small":
        file_path = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_Books_small.json"
        out_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_amazonbooks.txt"
    else:
        raise KeyError("Unknown dataset")

    # Set up the endpoint and your API key
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": "", # please fill your openaiapi key here
        "Content-Type": "application/json",
        "User-Agent": "OpenAI Python"
    }

    with open(file_path, 'r', encoding='utf8') as f:
        results = json.load(f)

    extra_added = "Please give the reasons in the format of list without explaining and prevent any possible data leakage, for example do not show the movie name"
    prompt_dict = {}
    for user_idx, value in results.items():
        for item_idx, item_prompt in value["prompt"].items():
            special_key = user_idx + ";" + item_idx
            prompt = item_prompt[0] + extra_added
            prompt = prompt.replace("<s>[INST]", "").replace("[/INST]", "")
            prompt_dict[special_key] = prompt


    already_done = set()
    with open(out_file_name, 'r', encoding='utf8') as f:
        for line in f:
            all_data = json.loads(line)
            if "union_id" in all_data:
                already_done.add(all_data["union_id"])

    while len(already_done) < len(prompt_dict):
        try:
            print("already_done", len(already_done))
            added_writter = open(out_file_name, 'a', encoding='utf8')
            for special_key, prompt in tqdm(prompt_dict.items()):
                if special_key in already_done:
                    continue
                results = get_response(prompt)
                if results != "Error":
                    added_writter.write(json.dumps(
                        {"union_id": special_key, "reasons": results}) + "\n")
                else:
                    time.sleep(10)
            added_writter.close()
            with open(out_file_name, 'r', encoding='utf8') as f:
                for line in f:
                    all_data = json.loads(line)
                    if "union_id" in all_data:
                        already_done.add(all_data["union_id"])
        except KeyboardInterrupt:
            break
        except:
            continue
    print("already_done", len(already_done))
