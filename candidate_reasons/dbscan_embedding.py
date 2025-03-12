import nmslib
import cuml
import torch
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# please change the file 
parser = ArgumentParser()
parser.add_argument("--reasons_file_path", type=str, default="candidate_reasons/raw_generated_reasons/llama2_7b-golden_reason_train_set_sep.json")
parser.add_argument("--cache_path", type=str, default="candidate_reasons/results_cache/")
parser.add_argument("--dataset", type=str, default="movielens")
args = parser.parse_args()
file_path = args.reasons_file_path
appendix = file_path.split("/")[-1].split("-")[0]+"_"+args.dataset

saved_file = f"cluster_results_MiniLM_UMAP20{appendix}"
saved_path = args.cache_path
saved_file_path = f"{saved_path}/{saved_file}.pkl"

if os.path.exists(saved_file_path):
    print(f"Loaded cache at {saved_file_path}")
    saved_results = pickle.load(open(saved_file_path, "rb"))
else:
    with open(file_path, "r", encoding='utf8') as f:
        reasons_results = json.load(f)
    print(len(reasons_results))

    ui_id2reason_id, reason_id2ui_id, reasons_idx = {}, {}, 0
    user_id_list, item_id_list, reasons_text = [], [], []
    for user_id, user_features in reasons_results.items():
        for item_id, item_prompts in user_features['prompt'].items():
            reasons_list = [re.sub(r'^\d+\.\s', '', i, flags=re.MULTILINE)
                            for i in item_prompts[-1].strip().split("\n")]
            for per_reasons in reasons_list:
                if len(per_reasons) <= 5:
                    continue
                user_id_list.append(user_id)
                item_id_list.append(item_id)
                reasons_text.append(per_reasons)
    assert len(user_id_list) == len(reasons_text)
    assert len(item_id_list) == len(reasons_text)
    print(len(reasons_text))

    model = SentenceTransformer(
        './data/pretrained_model/all-MiniLM-L6-v2').to("cuda:1")
    embeddings = model.encode(reasons_text, show_progress_bar=True)

    umap = cuml.manifold.UMAP(n_components=20, n_neighbors=15,
                              min_dist=0.0, random_state=12)
    reduced_data = umap.fit_transform(embeddings)
    print("UMAP done")

    print("start HDBSCAN")
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=5, metric='euclidean', prediction_data=True)
    clusterer.fit(reduced_data)
    print("clusterer done")

    print(pd.Series(clusterer.labels_).value_counts())

    print(f"Saved at {saved_file_path}")
    with open(saved_file_path, "wb") as f:
        saved_results = {"labels": clusterer.labels_, "text": reasons_text,
                         "user": user_id_list, "item": item_id_list}
        pickle.dump(saved_results, f)
    print("\n".join(random.sample(reasons_text, 5)))


label_counts = pd.Series(saved_results['labels']).value_counts()
label_counts.plot(kind='bar', color='skyblue', logy=True)
plt.title('Cluster Label Distribution')
plt.xlabel('Cluster Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # 标签水平显示
plt.savefig('example.png')

datasets_path = f"reasons_cf_datasets/{saved_file}"
if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)
user2reasons_text, item2reasons_text = {}, {}
output_user_id, output_item_id, output_reasons_id = [], [], []
for user_id, item_id, r_text, reasons_label in zip(saved_results["user"], saved_results["item"], saved_results["text"], saved_results['labels']):
    if reasons_label == -1:
        continue
    if user_id not in user2reasons_text.keys():
        user2reasons_text[user_id] = {}
    if item_id not in item2reasons_text.keys():
        item2reasons_text[item_id] = {}

    if reasons_label+1 not in user2reasons_text[user_id].keys():
        user2reasons_text[user_id][reasons_label+1] = []
    if reasons_label+1 not in item2reasons_text[item_id].keys():
        item2reasons_text[item_id][reasons_label+1] = []

    user2reasons_text[user_id][reasons_label+1].append(r_text)
    item2reasons_text[item_id][reasons_label+1].append(r_text)

    output_user_id.append(user_id)
    output_item_id.append(item_id)
    output_reasons_id.append(reasons_label+1)
assert len(output_user_id) == len(output_item_id)
assert len(output_user_id) == len(output_reasons_id)
print(len(output_user_id))

inter_data = {"user_id:token": output_user_id,
              "item_id:token": output_item_id, "reasons_id:token": output_reasons_id}
inter_data = pd.DataFrame(inter_data)
inter_data.to_csv(f"{datasets_path}/{saved_file}.inter", index=False, sep='\t')

with open(f"{datasets_path}/user_item2reasonsid_text.pkl", 'wb') as f:
    pickle.dump({"user": user2reasons_text,
                "item": item2reasons_text}, f)
print(f"new datasets saved at {datasets_path}")

