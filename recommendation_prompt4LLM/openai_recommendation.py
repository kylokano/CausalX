import requests
import json
import time
import os
from tqdm import tqdm

# Set up the endpoint and your API key
endpoint = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": "",
    "Content-Type": "application/json",
    "User-Agent": "OpenAI Python"
}

def get_response(prompt):
    # Set up the payload for chat-based models
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role":"user","content":prompt}],
        "max_tokens":500,
        "n":1,
        "temperature":0,
    }

    # Send the POST request
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    # Extract and print the response
    try:
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']
    except:
        assistant_message = "Error"
        print(response_data)
    # print(assistant_message)
    return assistant_message


files_dir = "recommendation_prompt4LLM/with_reasons_output/"
file_names = os.listdir(files_dir)

for per_name in file_names:
    file_path = files_dir + per_name
    out_file_name = "openai_recommendation_" + per_name
    out_file_name = "recommendation_prompt4LLM/with_reasons_results/" + out_file_name
    if not os.path.exists(out_file_name):
        with open(out_file_name, 'w', encoding='utf8') as f:
            f.write("")

    # file_path = "get_reasons_with_openai_amazoncd.json"
    # out_file_name = "openai_reasons_amazoncd.txt"
    with open(file_path, 'r', encoding='utf8') as f:
        results = json.load(f)

    prompt_dict = {}
    for user_idx, value in results.items():
        prompt = value[3]
        special_key  = user_idx
        prompt = prompt.replace("<s>[INST]", "").replace("[/INST]", "")
        prompt_dict[special_key] = prompt + "Please only provide the name."


    already_done = set()
    with open(out_file_name, 'r', encoding='utf8') as f:
        for line in f:
            all_data = json.loads(line)
            if "user_id" in all_data:
                already_done.add(all_data["user_id"])

    try_number = 0
    while len(already_done) < len(prompt_dict):
        try:
            print("already_done", len(already_done))
            added_writter = open(out_file_name, 'a', encoding='utf8')
            for special_key, prompt in tqdm(prompt_dict.items()):
                if special_key in already_done:             
                    continue 
                results = get_response(prompt)
                if results != "Error":
                    added_writter.write(json.dumps({"user_id": special_key, "Recommendations": results}) + "\n")
                    try_number+=1
                    # if try_number == 5:
                    #     raise KeyboardInterrupt
                else:
                    time.sleep(10)
            added_writter.close()
            with open(out_file_name, 'r', encoding='utf8') as f:
                for line in f:
                    all_data = json.loads(line)
                    if "user_id" in all_data:
                        already_done.add(all_data["user_id"])
        except KeyboardInterrupt:
            break
        except:
            
            continue
    print("already_done", len(already_done))