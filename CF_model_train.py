from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import *
from recbole.model.general_recommender import *
from recbole.model.knowledge_aware_recommender import *
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.quick_start import run_recbole
from sympy import per
# from SimpleX_reasons import SimpleX4Reasons
from rs_models.NCF_reasons import NeuMFReasons
# from rs_models.NCF_reasons_pairwise import NeuMFReasonsPairwise

from load_datasets import load_recbole_datasets
import copy
import torch

if __name__ == '__main__':
    # configurations initialization
    print("Data Loading for RecBole...")
    # dataset = "amazonbooks"
    # if dataset == "movielens":
    #     config = Config(model='NeuMF', dataset='cluster_results_MiniLM_UMAP20_candidate_movielens', config_file_list=[
    #         "config/CF_reasons.yaml", "config/SimpleX.yaml"])
    # elif dataset == "amazoncd":
    #     config = Config(model='NeuMF', dataset='cluster_results_MiniLM_UMAP20_candidate_amazoncd', config_file_list=[
    #         "config/CF_reasons.yaml", "config/SimpleX.yaml"])
    # elif dataset == "amazonbooks":
    #     config = Config(model='NeuMF', dataset='cluster_results_MiniLM_UMAP20_candidate_amazonbooks', config_file_list=[
    #         "config/CF_reasons.yaml", "config/SimpleX.yaml"])
    # else:
    #     raise ValueError(f"dataset {dataset} not supported")
    
    # 如果命令行传入了dataset参数，自动添加前缀
    config = Config(model='NeuMF', config_file_list=["config/CF_reasons.yaml", "config/SimpleX.yaml"])
    # 验证是否正确
    print(config["dataset"])
    
    if "is_pairwise" not in config or config["is_pairwise"]:
        raise KeyError("is_pairwise is discarded, it must be set to False")
    config.model = "NeuMFReasons"
    init_logger(config)
    logger = getLogger()
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # write config info into log
    logger.info(config)

    train_data, valid_data, test_data = load_recbole_datasets(logger, config)

    model_class = eval(config.model)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

