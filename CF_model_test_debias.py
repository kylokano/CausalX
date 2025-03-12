from logging import getLogger
from lightgbm import train
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer
from sklearn.conftest import dataset_fetchers

from load_datasets import load_recbole_datasets, load_context_data, seqdataloader_convert2prompt

from vllm import LLM
from vllm import SamplingParams

import os
import torch
import json
from tqdm import tqdm
import numpy as np

from llm_evaluation import load_results, VirtueLLM
from collections import defaultdict
from rs_models.NCF_reasons_popbias import NeuMFReasonsPop

from recbole.model.general_recommender import NGCF, SimpleX
import argparse

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazonbooks")
    parser.add_argument("--debias_coeffiecient", type=float, default=0.1)
    parser.add_argument("--best_model_path", type=str, default="saved/NeuMF-Nov-17-2024_00-30-43.pth")
    args = parser.parse_args()
    
    if args.dataset == "ml-latest-small":
        dataset = "movielens"
    elif args.dataset == "Amazon_CDs_and_Vinyl_small":
        dataset = "amazoncd"
    elif args.dataset == "Amazon_Books_small":
        dataset = "amazonbooks"
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    debias_coeffiecient = args.debias_coeffiecient
    best_model_path = args.best_model_path
    
    config = Config(model='NeuMF', dataset="cluster_results_MiniLM_UMAP20_openai_"+dataset, config_file_list=["config/CF_reasons.yaml", "config/SimpleX.yaml"])
    print(config["dataset"])
    config["debias_coeffiecient"] = debias_coeffiecient
    debias_coeffiecient = config["debias_coeffiecient"]
    print("Debias Coeffiecient", debias_coeffiecient)
    # dataset = config["dataset"].split("_")[-1]
    # raise ValueError("Not support debias")
    
    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    logger.info(config)
    cf_train_data, cf_valid_data, cf_test_data = load_recbole_datasets(
        logger, config)
    cf_field2tokenid = cf_train_data.dataset.field2token_id
    cf_field2id_token = cf_train_data.dataset.field2id_token


    trained_cf_model = torch.load(best_model_path)
    if trained_cf_model["config"]["is_pairwise"]:
        raise NotImplementedError("Not support pairwise model")
    else:
        print("PointWise Training")
        config.model = "NeuMFReasonsPop"
    model_class = eval(config.model)
    model = model_class(trained_cf_model["config"], cf_train_data.dataset)
    model.load_state_dict(trained_cf_model["state_dict"])
    
    
    if dataset == "movielens":
        config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
            "config/sequential_ml.yaml", "config/LLM.yaml"])
    elif dataset == "amazoncd":
        config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
    elif dataset == "amazonbooks":
        config = Config(model='BERT4Rec', dataset='Amazon_Books_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
    else:
        raise ValueError(f"dataset {dataset} not supported")
    
    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    logger.info(config)

    train_data, valid_data, test_data = load_recbole_datasets(logger, config)
    model_item_id2raw_dataset_itemid, model_user_id2raw_dataset_userid = {}, {}
    for raw_dataset_itemid, model_itemid in test_data._dataset.field2token_id["item_id"].items():
        model_item_id2raw_dataset_itemid[model_itemid] = raw_dataset_itemid
    for raw_dataset_userid, model_user_id in test_data._dataset.field2token_id["user_id"].items():
        model_user_id2raw_dataset_userid[model_user_id] = raw_dataset_userid
    user_for_reasons = defaultdict(list)
    max_user_id = 0
    for cur_data, idx_list, positive_u, positive_i in test_data:
        # 遍历数据，填充字典
        for index, (user, item, label, history) in enumerate(zip(cur_data["user_id"], cur_data["item_id"], cur_data["label"], cur_data["item_id_list"])):
            # if label == 1:
            if user.item() > max_user_id:
                max_user_id = user.item()
            user_for_reasons[model_user_id2raw_dataset_userid[user.item()]].append(
                model_item_id2raw_dataset_itemid[item.item()])
            # if dataset_part == "amazoncd":
            #     user_for_reasons[model_user_id2raw_dataset_userid[user.item()]].append(
            #         model_item_id2raw_dataset_itemid[item.item()])
            # else:
            #     user_for_reasons[user.item()].append(
            #         model_item_id2raw_dataset_itemid[item.item()])
            # else:
            #     user_samples[user.item()][1].append(
            #         model_item_id2raw_dataset_itemid[item.item()])  # 添加到负样本列表

    model = model.to("cuda")
    user_test2reasons = {}
    all_item, not_ava_item = 0, 0
    with torch.no_grad():
        for per_user, candidate_item_lists in tqdm(user_for_reasons.items()):
            all_reasons = []
            if str(per_user) not in cf_field2tokenid["user_id"]:
                # print("user", per_user)
                user_test2reasons[per_user] = [[-1], [-1]]
                continue
            input_user = cf_field2tokenid["user_id"][str(per_user)]
            input_user = torch.tensor(
                [input_user], dtype=torch.int64).to("cuda")
            for per_item in candidate_item_lists:
                all_item += 1
                if per_item not in cf_field2tokenid["item_id"]:
                    # not_ava_item += 1
                    # print("item", per_item)
                    continue
                input_item = cf_field2tokenid["item_id"][per_item]
                input_item = torch.tensor(
                    [input_item], dtype=torch.int64).to("cuda")
                part_results = model.full_reasons_predict(
                    {"user_id": input_user, "item_id": input_item})
                part_results[0, :] = -np.inf
                all_reasons.append(part_results)
            reasons_num = all_reasons[0].shape[0]
            all_reasons = torch.cat(all_reasons, dim=0).squeeze(1)
            topk_values, topk_index = torch.topk(all_reasons, 50, dim=0)
            topk_index = topk_index.to("cpu").tolist()
            topk_index = [i % reasons_num for i in topk_index]
            topk_index = [cf_field2id_token["reasons_id"][i]
                        for i in topk_index]
            user_test2reasons[per_user] = [
                topk_index, topk_values.to("cpu").tolist()]
    with open(f"reasons_cf_datasets/cluster_results_MiniLM_UMAP20_openai_{dataset}/user2reasons_{debias_coeffiecient}.json", 'w', encoding='utf8') as f:
        json.dump(user_test2reasons, f)
