from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer

from load_datasets import load_recbole_datasets, load_context_data, seqdataloader_convert2prompt

import json
import pandas as pd
import os
import pickle
import re
from shutil import copyfile

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml-latest-small")
    parser.add_argument("--feature_type", type=str, default="only_name")
    parser.add_argument("--topk_reasons", type=int, default=10)
    parser.add_argument("--debias_coffecient", type=float, default=1)
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "ml-latest-small":
        dataset = "movielens"
    elif dataset == "Amazon_CDs_and_Vinyl_small":
        dataset = "amazoncd"
    elif dataset == "Amazon_Books_small":
        dataset = "amazonbooks"
    feature_type = args.feature_type
    debias_coffecient = args.debias_coffecient
    topk = args.topk_reasons
    
    # topk_reasons = [1, 3, 5, 10]
    # debias_coffecient = [0.1, 0.5, 1, 5, 10]
    reason_cf_dataset_name = f"cluster_results_MiniLM_UMAP20_openai_{dataset}"
    
    print("Data Loading for RecBole...")
    if dataset == "amazoncd":
        config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
            "/home/nfs03/ligr/llmrs/config/sequential_ml_amazon.yaml", "/home/nfs03/ligr/llmrs/config/LLM.yaml"])
    elif dataset == "amazonbooks":
        config = Config(model='BERT4Rec', dataset='Amazon_Books_small', config_file_list=[
            "/home/nfs03/ligr/llmrs/config/sequential_ml_amazon.yaml", "/home/nfs03/ligr/llmrs/config/LLM.yaml"])
    else:
        config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
            "/home/nfs03/ligr/llmrs/config/sequential_ml.yaml", "/home/nfs03/ligr/llmrs/config/LLM.yaml"])
    
    config["feature_type"] = feature_type
    config["cf_dataset_name"] = reason_cf_dataset_name
    config["debias_coffecient"] = debias_coffecient
    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    logger.info(config)

    train_data, valid_data, test_data = load_recbole_datasets(logger, config)
    user_context, item_context, user_item_inter = load_context_data(
        logger, config)

    USER_ID = config["USER_ID_FIELD"]
    ITEM_ID = config["ITEM_ID_FIELD"]
    n_users = train_data.dataset.num(USER_ID)
    n_items = train_data.dataset.num(ITEM_ID)
    logger.info(f"user_number {n_users}, item_number {n_items}")

    user_dict = seqdataloader_convert2prompt(
        user_context, item_context, test_data, config["prompt_type"], logger, config, test_data._dataset.field2token_id["item_id"], test_data._dataset.field2token_id["user_id"])

    # save the recommendaiton prompt
    dataset_name = config["dataset"]
    item_features = config["item_features"]
    print(f"The prompt of {dataset_name} {item_features}")
    with open(f"recommendation_prompt4LLM/{dataset_name}_{item_features}_prompt.json", 'w', encoding='utf8') as f:
        json.dump(user_dict, f)
    
    output_path = "recommendation_prompt4LLM/with_reasons_output"
    if dataset == "amazoncd":
        reasons_predict_path = "reasons_cf_datasets/cluster_results_MiniLM_UMAP20openai_amazoncd"
        raw_prompt_file = f"recommendation_prompt4LLM/Amazon_CDs_and_Vinyl_small_{feature_type}_prompt.json"
    elif dataset == "amazonbooks":
        reasons_predict_path = "reasons_cf_datasets/cluster_results_MiniLM_UMAP20_openai_amazonbooks"
        raw_prompt_file = f"recommendation_prompt4LLM/Amazon_Books_small_{feature_type}_prompt.json"
    else:
        reasons_predict_path = "reasons_cf_datasets/cluster_results_MiniLM_UMAP20_openai_movielens"
        raw_prompt_file = f"recommendation_prompt4LLM/ml-latest-small_{feature_type}_prompt.json"
    per_file = f"user2reasons_{debias_coffecient}.json"
    raw_data = user_dict

    with open(reasons_predict_path + "/" + "user_item2reasonsid_text.pkl", 'rb') as f:
        user_reasonid2text = pickle.load(f)
    reasonsid2text = {}
    for user_id, user_reasons in user_reasonid2text['user'].items():
        for per_reasons_id in user_reasons:
            if per_reasons_id not in reasonsid2text:
                reasonsid2text[per_reasons_id] = user_reasons[per_reasons_id]
            reasonsid2text[per_reasons_id] = reasonsid2text[per_reasons_id] + \
                user_reasons[per_reasons_id]


    with open(reasons_predict_path+"/"+per_file, 'r', encoding='utf8') as f:
        userid2predicted_reasons = json.load(f)
    withreasons_dict = {}
    for user_id, value in raw_data.items():
        if user_id not in userid2predicted_reasons:
            print(user_id)
            continue
        predicted_reasons_id = userid2predicted_reasons[user_id][0][:topk]
        predicted_reasons_text = []
        for i in predicted_reasons_id:
            if user_id in user_reasonid2text['user'] and i in user_reasonid2text['user'][user_id]:
                predicted_reasons_text.append(
                    user_reasonid2text['user'][user_id][i][0])
            else:
                if int(i) == -1:
                    continue
                predicted_reasons_text.append(reasonsid2text[int(i)][0])
        predicted_reasons_text = [
            re.sub(r'^\d+\.\s*', '', i) for i in predicted_reasons_text]
        if dataset == "amazoncd":
            raw_prompt = value[3]
            raw_prompt = raw_prompt.split("Please rank the candidate CDs")
            raw_prompt = raw_prompt[0] + "and the reasons for potentially choosing to watch, listed as: " + \
                ",\n".join(predicted_reasons_text) + \
                "\nPlease rank the candidate CDs" + raw_prompt[1]
        elif dataset == "movielens":
            raw_prompt = value[3]
            raw_prompt = raw_prompt.split(
                "please rank the candidate movies")
            raw_prompt = raw_prompt[0] + "and the reasons for potentially choosing to watch, listed as: " + \
                ",\n".join(predicted_reasons_text) + \
                "\nPlease rank the candidate movies" + raw_prompt[1]
        elif dataset == "amazonbooks":
            raw_prompt = value[3]
            raw_prompt = raw_prompt.split(
                "Please rank the candidate books")
            raw_prompt = raw_prompt[0] + "and the reasons for potentially choosing to read, listed as: " + \
                ",\n".join(predicted_reasons_text) + \
                "\nPlease rank the candidate books" + raw_prompt[1]
        else:
            raise NotImplementedError
        withreasons_dict[user_id] = [
            value[0], value[1], value[2], raw_prompt]
    with open(output_path + "/" + f"{dataset}_" + per_file[:-5] + f"_{feature_type}_top{topk}.json", 'w', encoding='utf8') as f:
        json.dump(withreasons_dict, f)
    print("Down")



