from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer

from load_datasets import load_recbole_datasets, load_context_data, seqdataloader_convert2prompt

from vllm import LLM
from vllm import SamplingParams

import os
import torch
import json

from llm_evaluation import load_results, VirtueLLM

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
        "config/sequential_ml.yaml", "config/LLM.yaml"])
    # config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
    #     "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])

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

    if "debug" in config and config["debug"]:
        sample_keys = list(user_dict.keys())[:6]
        new_user_dict = {}
        for per_key in sample_keys:
            new_user_dict[per_key] = user_dict[per_key]
        user_dict = new_user_dict
        logger.info(f"Debug model with datasize {len(user_dict)}")

    if config["prompt_type"] == "get_golden_reason":
        output_results, input_prompt = [], []
        for key, value in user_dict.items():
            for prompt_value in value[3].values():
                output_results.append(prompt_value)
                input_prompt.append(prompt_value[0])
    else:
        input_prompt = [value[3] for value in user_dict.values()]
    logger.info("Total prompts for evaluation: {}".format(len(input_prompt)))
    logger.info("Load model from {}".format(config["model_name_or_path"]))

    saved_file = os.path.join("llm_results", config["exp_name"]+".json")
    if "70b" in config["model_name_or_path"]:
        logger.info("tensor_parallel_size")
        llm = LLM(model=config['model_name_or_path'], tokenizer=config["model_name_or_path"], dtype="half", trust_remote_code=True,
                  swap_space=32, tensor_parallel_size=4, gpu_memory_utilization=0.3)
    else:
        llm = LLM(model=config['model_name_or_path'], tokenizer=config["model_name_or_path"], dtype="half", trust_remote_code=True,
                  swap_space=64)
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=1024
    )

    with torch.no_grad():
        outputs = llm.generate(
            input_prompt,
            sampling_params
        )
    generated = [output.outputs[0].text for output in outputs]

    if config["prompt_type"] == "get_golden_reason":
        for res, gen_text in zip(output_results, generated):
            res.append(gen_text)
    else:
        for (key, value), gen_text in zip(user_dict.items(), generated):
            value.append(gen_text)

    with open(saved_file, 'w', encoding='utf8') as f:
        json.dump(user_dict, f)
    logger.info(f"Saved in {saved_file}")

    logger.info(f"Load from {saved_file}")
    with open(saved_file, 'r', encoding='utf8') as f:
        user_dict = json.load(f)

    pred_u, pred_i, value, pos_u, pos_i = load_results(
        user_dict, None, item_context, config,  logger, test_data._dataset.field2token_id["item_id"])

    trainer = Trainer(config, VirtueLLM())
    trainer.evaluate([pred_u, pred_i, value, pos_u, pos_i,
                     test_data._dataset.item_num], None)
