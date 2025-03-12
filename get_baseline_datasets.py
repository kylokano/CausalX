from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer

from load_datasets import load_recbole_datasets, load_context_data, traindata_get_reaons, seqdataloader_convert2prompt

from vllm import LLM
from vllm import SamplingParams

import os
import torch
import json

from llm_evaluation import load_results

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    # datasets_name = "ml-latest-small"
    # datasets_name = "Amazon_CDs_and_Vinyl_small"
    datasets_name = "Amazon_Books_small"

    if datasets_name == "ml-latest-small":
        config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
            "config/sequential_ml.yaml", "config/LLM.yaml"])
        baseline_data_path_dir = "baselines/baseline_datasets/ml-latest-small_b"
        
    elif datasets_name == "Amazon_CDs_and_Vinyl_small":
        config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
        baseline_data_path_dir = "baselines/baseline_datasets/Amazon_CDs_and_Vinyl_small_b"
    elif datasets_name == "Amazon_Books_small":
        config = Config(model='BERT4Rec', dataset='Amazon_Books_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
        baseline_data_path_dir = "baselines/baseline_datasets/Amazon_Books_small_b"
    else:
        raise KeyError("Unknown dataset")

    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    config["prompt_type"] = "get_golden_reasons"
    logger.info(config)
    # # config["item_features"] = "openai"
    # config["item_features"] = "openai"

    train_data, valid_data, test_data = load_recbole_datasets(logger, config)

    user_context, item_context, user_item_inter = load_context_data(
        logger, config)

    user_dict = traindata_get_reaons(
        user_context, item_context, train_data, logger, config, test_data._dataset.field2token_id["user_id"], test_data._dataset.field2token_id["item_id"])
    
    config["prompt_type"] = "zero_shot"
    test_user_dict = seqdataloader_convert2prompt(
        user_context, item_context, test_data, config["prompt_type"], logger, config, test_data._dataset.field2token_id["item_id"], test_data._dataset.field2token_id["user_id"])
    
    baselines_dataset_name = datasets_name + "_b"
    
    print("saved at ",os.path.join(baseline_data_path_dir, baselines_dataset_name))
    with open(os.path.join(baseline_data_path_dir, baselines_dataset_name + ".inter"), 'w', encoding='utf8') as f:
        f.write("user_id:token\titem_id:token\n")
        for user_id, user_features in user_dict.items():
            all_item_list = user_features["pos_item"] + user_features["his_item"]
            for per_item in all_item_list:
                if per_item == "[PAD]":
                    continue
                f.write(f"{user_id}\t{per_item}\n")
    
    
    with open(os.path.join(baseline_data_path_dir, baselines_dataset_name + "_test.inter"), 'w', encoding='utf8') as f:
        f.write("user_id:token\tpos_item_id:token\tneg_item_id:token_seq\this_item_id:token_seq\n")
        for user_id, user_features in test_user_dict.items():
            user_id, pos_item, neg_item, his_item = str(user_id), ",".join(map(str, user_features[0])), ",".join(map(str, user_features[1])), ",".join(map(str, user_features[2]))
            f.write(f"{user_id}\t{pos_item}\t{neg_item}\t{his_item}\n")