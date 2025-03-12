import os
import json

raw_file_path = "recommendation_prompt4LLM/with_reasons_output/"
raw_files = os.listdir(raw_file_path)
results_path = "recommendation_prompt4LLM/with_reasons_results/"
output_path = "recommendation_prompt4LLM/with_reasons_combinations/"

for per_file in raw_files:
    if "amazonbooks" in per_file:
        continue
    with open(raw_file_path + per_file, 'r', encoding='utf8') as f:
        data = json.load(f)
        
    results_dict = {}
    with open(results_path + "openai_recommendation_" + per_file, 'r', encoding='utf8') as f:
    # with open(results_path +  per_file, 'r', encoding='utf8') as f:
        for line in f:
            if len(line) <5:
                continue
            line_data = json.loads(line)
            results_dict[line_data['user_id']] = line_data['Recommendations']
    
    for user_id, value in results_dict.items():
        data[user_id].append(results_dict[user_id])
    
    with open(output_path + "openai_" + per_file, 'w', encoding='utf8') as f:
        json.dump(data, f)
