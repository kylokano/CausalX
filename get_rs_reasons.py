from logging import getLogger
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer

from load_datasets import load_recbole_datasets, load_context_data, traindata_get_reaons

from vllm import LLM
from vllm import SamplingParams

import os
import torch
import json

from llm_evaluation import load_results

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    datasets_name = "ml-latest-small"

    if datasets_name == "ml-latest-small":
        config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
            "config/sequential_ml.yaml", "config/LLM.yaml"])
    elif datasets_name == "Amazon_CDs_and_Vinyl_small":
        config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
            "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
    else:
        raise KeyError("Unknown dataset")

    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    config["prompt_type"] = "get_golden_reasons"
    logger.info(config)

    train_data, valid_data, test_data = load_recbole_datasets(logger, config)

    user_context, item_context, user_item_inter = load_context_data(
        logger, config)

    user_dict = traindata_get_reaons(
        user_context, item_context, train_data, logger, config, test_data._dataset.field2token_id["item_id"])
    output_results, input_prompt = [], []
    for key, value in user_dict.items():
        for prompt_value in value["prompt"].values():
            output_results.append(prompt_value)
            input_prompt.append(prompt_value[0])
    logger.info("Total prompts for evaluation: {}".format(len(input_prompt)))
    saved_file = os.path.join(
        "candidate_reasons/raw_generated_reasons", config["exp_name"]+".json")
    if "13b" in config['model_name_or_path']:
        llm = LLM(model=config['model_name_or_path'], tokenizer=config["model_name_or_path"], dtype="half", trust_remote_code=True,
                  swap_space=32, tensor_parallel_size=2)
    else:
        llm = LLM(model=config['model_name_or_path'], tokenizer=config["model_name_or_path"], dtype="half", trust_remote_code=True,
                  swap_space=64, gpu_memory_utilization=1)
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=512
    )

    with torch.no_grad():
        outputs = llm.generate(
            input_prompt,
            sampling_params
        )
    generated = [output.outputs[0].text for output in outputs]

    for res, gen_text in zip(output_results, generated):
        res.append(gen_text)

    with open(saved_file, 'w', encoding='utf8') as f:
        json.dump(user_dict, f)
    logger.info(f"Saved in {saved_file}")
