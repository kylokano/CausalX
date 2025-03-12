import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ml-latest-small")
args = parser.parse_args()
dataset = args.dataset

if dataset == "ml-latest-small":
    dataset = "movielens"
elif dataset == "Amazon_CDs_and_Vinyl_small":
    dataset = "amazoncd"
elif dataset == "Amazon_Books_small":
    dataset = "amazonbooks"
else:
    raise ValueError(f"dataset {dataset} not supported")

file_name = f"cluster_results_MiniLM_UMAP20_openai_{dataset}"
file_path = f"reasons_cf_datasets/{file_name}/{file_name}.inter"
data = pd.read_csv(file_path, sep="\t")
data.head()

frequency = data["reasons_id:token"].value_counts()
frequency = frequency.to_dict()

# %%
from math import log


total_count = sum(frequency.values())
for reasons_id, reasons_count in frequency.items():
    frequency[reasons_id] = log(reasons_count + 1)

# standard
max_num = max(frequency.values())
min_num = min(frequency.values())
for reasons_id, reasons_count in frequency.items():
    frequency[reasons_id] = (frequency[reasons_id] -
                             min_num) / (max_num - min_num)
# frequency

# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 5))
# plt.hist(frequency.values(), bins=100)
# plt.xlabel("log(frequency)")
# plt.ylabel("count")
# plt.title("log(frequency) distribution")
# plt.show()

import pickle
with open(f"rs_models/popularity/{file_name}_logfrequency.pkl", 'wb') as f:
    pickle.dump(frequency, f)
