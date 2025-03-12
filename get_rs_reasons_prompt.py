from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer

from load_datasets import load_recbole_datasets, load_context_data, traindata_get_reasons, seqdataloader_convert2prompt

from vllm import LLM
from vllm import SamplingParams

import os
import torch
import json

from llm_evaluation import load_results
import argparse

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    # datasets_name = "ml-latest-small"
    # datasets_name = "Amazon_CDs_and_Vinyl_small"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Amazon_Books_small")
    args = parser.parse_args()
    datasets_name = args.dataset

    if datasets_name == "ml-latest-small":
        config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
            "config/sequential_ml.yaml", "config/LLM.yaml"])
    elif datasets_name == "Amazon_CDs_and_Vinyl_small":
        config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
    elif datasets_name == "Amazon_Books_small":
        config = Config(model='BERT4Rec', dataset='Amazon_Books_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
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

    user_dict = traindata_get_reasons(
        user_context, item_context, train_data, logger, config, test_data._dataset.field2token_id["user_id"], test_data._dataset.field2token_id["item_id"])
    
    with open(f"candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_{datasets_name}.json", 'w', encoding='utf8') as f:
        json.dump(user_dict, f)
    
    
        