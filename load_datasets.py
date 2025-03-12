import pickle
# from reason2list import get_reasons, get_preend_tokens
from collections import defaultdict
import json
import random
import os
import pandas as pd
from recbole.utils import init_seed, init_logger
from recbole.trainer import Trainer
from recbole.model.sequential_recommender import BERT4Rec
from recbole.model.general_recommender import BPR, LightGCN
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from logging import getLogger
import re

def get_special_tokens(config):
    if "llama-3" in config["model_name_or_path"]:
        pre_tokens = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_tokens = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "llama-2" in config["model_name_or_path"] or "Mistral" in config["model_name_or_path"]:
        pre_tokens = "<s>[INST] "
        end_tokens = " [/INST]"
    else:
        pre_tokens = ""
        end_tokens = ""
    return pre_tokens, end_tokens


def get_reasons(text_input):
    text_input = text_input.replace(
        "Detailed explanation", "").replace("Key phrase", "")
    text_input = re.sub(r'^[\W_]+', '', text_input, flags=re.MULTILINE)
    return text_input



def load_recbole_datasets(logger, config):
    # init random seed, to makesure all seperates are same
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return train_data, valid_data, test_data


def load_context_data(logger, config):
    dataset_name = config["dataset"]
    logger.info("Load context data for dataset: "+dataset_name)
    data_path = config.data_path
    user_feature = pd.read_csv(os.path.join(
        data_path, dataset_name+".user"), sep="\t")
    item_feature = pd.read_csv(os.path.join(
        data_path, dataset_name+".item"), sep="\t")
    user_context, item_context = {}, {}

    def convert2dict(df: pd.DataFrame):
        result_dict = {}
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():
            # 使用item_id列的值作为键，其余列组成的字典作为值
            if "user_id:token" in row:
                key = row['user_id:token']
            else:
                key = row['item_id:token']
            value = {col: row[col] for col in df.columns if col != 'Name'}
            result_dict[key] = value
        return result_dict
    user_context, item_context = convert2dict(
        user_feature), convert2dict(item_feature)

    logger.info("Load user interaction data for dataset: "+dataset_name)
    inter_data = pd.read_csv(os.path.join(
        data_path, dataset_name+".inter"), sep='\t')

    user_item_inter = defaultdict(list)
    print(inter_data.columns)
    for index, row in inter_data.iterrows():
        # user_id:token	item_id:token	rating:float	timestamp:float
        if "rating:float" in inter_data.columns:
            user_id, item_id, rating, time = row["user_id:token"], row[
                "item_id:token"], row["rating:float"], row["timestamp:float"]
            user_item_inter[user_id].append((time, item_id, rating))
        else:
            user_id, item_id, time = row["user_id:token"], row["item_id:token"], row["timestamp:float"]
            user_item_inter[user_id].append((time, item_id))
    for key, value in user_item_inter.items():
        user_item_inter[key] = sorted(value, key=lambda x: x[0])

    return user_context, item_context, user_item_inter

# todo 将bole dataset 原封不动转化为LLM可以理解的内容


def itemid2context(item_context, item_id_seq, config):
    if "ml-latest-small" in config["dataset"]:
        # item_id:token	movie_title:token_seq	release_year:token	class:token_seq	intro:token_seq	directors:token_seq	writers:token_seq	starts:token_seq
        item_list = []
        for idx in item_id_seq:
            if idx == "[PAD]":
                continue
            idx = int(idx)
            context = item_context[idx]
            if config["item_features"] == "all_features":
                item_list.append(
                    f"\"'{context['movie_title:token_seq']}', which was released in {context['release_year:token']}, belongs to the genre {context['class:token_seq']}, was directed by {context['directors:token_seq']} and have stars {context['starts:token_seq']}\"")
            elif config["item_features"] == "base_features":
                item_list.append(
                    f"\"'{context['movie_title:token_seq']}', which was released in {context['release_year:token']}, belongs to the genre {context['class:token_seq']}\"")
            elif config["item_features"] == "only_name":
                item_list.append(f"'{context['movie_title:token_seq']}'")
            elif config["item_features"] == "openai":
                item_list.append(
                    f"'{context['movie_title:token_seq']}' ({context['release_year:token']}), an {context['class:token_seq']}")
            else:
                raise KeyError("The item_features is not supported yet.")
    elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
        item_list = []
        for idx in item_id_seq:
            if idx == "[PAD]":
                continue
            # idx = int(idx)
            context = item_context[idx]
            # item_id:token	title:token	categories:token_seq	brand:token	sales_type:token	sales_rank:float
            if config["item_features"] == "base_features" or config["item_features"] == "openai":
                item_list.append(
                    f"\"'{context['title:token']}', which was played by band {context['brand:token']}, belongs to {context['categories:token_seq']}\"")
            elif config["item_features"] == "only_name":
                item_list.append(f"'{context['title:token']}'")
            else:
                raise KeyError("The item_features is not supported yet.")
    elif config["dataset"] == "Amazon_Books_small":
        item_list = []
        for idx in item_id_seq:
            if idx == "[PAD]":
                continue
            context = item_context[idx]
            # item_id:token	sales_type:token	sales_rank:float	categories:token_seq	title:token	price:float	brand:token
            if config["item_features"] == "base_features" or config["item_features"] == "openai":
                item_list.append(
                    f"\"'{context['title:token']}', sells for {context['price:float']}\"")
            elif config["item_features"] == "only_name":
                item_list.append(f"'{context['title:token']}'")
            else:
                raise KeyError("The item_features is not supported yet.")
    else:
        raise KeyError("The dataset is not supported yet.")
    return "\n".join(item_list)+"\n"


def traindata_get_reasons(user_context, item_context, seq_dataloader, logger, config, raw_dataset_userid2model_userid, raw_dataset_itemid2model_itemid):
    model_item_id2raw_dataset_itemid, model_user_id2raw_dataset_userid = {}, {}
    for raw_dataset_itemid, model_itemid in raw_dataset_itemid2model_itemid.items():
        model_item_id2raw_dataset_itemid[model_itemid] = raw_dataset_itemid
    
    for raw_dataset_userid, model_useid in raw_dataset_userid2model_userid.items():
        model_user_id2raw_dataset_userid[model_useid] = raw_dataset_userid

    # pre_tokens = "<s>[INST] "
    # end_tokens = " [/INST]"
    pre_tokens, end_tokens = get_special_tokens(config)
    user_samples = defaultdict(lambda: {"pos_item": [], "his_item": []})

    for cur_data in seq_dataloader:
        # 遍历数据，填充字典
        for index, (user, item, history) in enumerate(zip(cur_data["user_id"], cur_data["item_id"], cur_data["item_id_list"])):
            user_samples[model_user_id2raw_dataset_userid[user.item()]]["pos_item"].append(
                model_item_id2raw_dataset_itemid[item.item()])  # 添加到正样本列表
            user_samples[model_user_id2raw_dataset_userid[user.item()]]["his_item"] = [model_item_id2raw_dataset_itemid[i]
                                                     for i in history.tolist()]

    for user_id, user_feature in user_samples.items():
        his_movie = itemid2context(
            item_context, user_feature["his_item"], config)
        prompt_dict = {}
        for pos_item in user_feature["pos_item"]:
            next_movie = itemid2context(
                item_context, [pos_item], config)
            # prompt = f"The user has previously watched the movie '{his_movie}' and subsequently choose to watch '{next_movie}'. Please analyze the user's viewing history and provide reasons for why they might have selected '{next_movie}' as their next movie to watch."

            if config["dataset"] == "ml-latest-small":
                prompt = f"The user has previously watched the movie:\n'{his_movie}\nPlease analyze the user's viewing history and provide reasons for why they might have selected '{next_movie}' as their next movie to watch."
            elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
                prompt = f"The user has previously bought CDs:\n'{his_movie}\nPlease analyze the user's listening history and provide reasons for why he select '{next_movie}' as his next CD to bought."
            elif config["dataset"] == "Amazon_Books_small":
                prompt = f"The user has previously bought books:\n'{his_movie}\nPlease analyze the user's reading history and provide reasons for why he select '{next_movie}' as his next book to buy."
            else:
                raise KeyError
            prompt_dict[pos_item] = [pre_tokens + prompt + end_tokens]
        user_feature["prompt"] = prompt_dict
    return user_samples


def load_reasons(logger, config):
    reason_file_path = config["reason_file_path"]
    logger.info(f"Load reasons from {reason_file_path}")
    reason_results = json.load(
        open(reason_file_path, "r", encoding='utf8'))
    user_id2reasons = {}
    for key, value in reason_results.values():
        key_reasons = value[4]
        user_id2reasons[key] = value[4]


def seqdataloader_convert2prompt(user_context, item_context, seq_dataloader, prompt_type, logger, config, raw_dataset_itemid2model_itemid, raw_dataset_userid2model_userid):
    """convert recbole seqnegsampleevaldataloader to prompt for LLM evaluation
    prompt_label: to change the prompt for different variant;
    """
    model_item_id2raw_dataset_itemid, model_user_id2raw_dataset_userid = {}, {}
    for raw_dataset_itemid, model_itemid in raw_dataset_itemid2model_itemid.items():
        model_item_id2raw_dataset_itemid[model_itemid] = raw_dataset_itemid
    for raw_dataset_userid, model_user_id in raw_dataset_userid2model_userid.items():
        model_user_id2raw_dataset_userid[model_user_id] = raw_dataset_userid

    # pre_tokens = "<s>[INST] "
    # end_tokens = " [/INST]"
    pre_tokens, end_tokens = get_special_tokens(config)
    user_samples = defaultdict(lambda: [[], [], []])

    for cur_data, idx_list, positive_u, positive_i in seq_dataloader:
        # 遍历数据，填充字典
        for index, (user, item, label, history) in enumerate(zip(cur_data["user_id"], cur_data["item_id"], cur_data["label"], cur_data["item_id_list"])):
            raw_user_id = model_user_id2raw_dataset_userid[user.item()]
            if label == 1:
                user_samples[raw_user_id][0].append(
                    model_item_id2raw_dataset_itemid[item.item()])  # 添加到正样本列表
            else:
                user_samples[raw_user_id][1].append(
                    model_item_id2raw_dataset_itemid[item.item()])  # 添加到负样本列表
            user_samples[raw_user_id][2] = [model_item_id2raw_dataset_itemid[i]
                                            for i in history.tolist()]

    # only for provied reasons
    if prompt_type == "random_train_reason":
        if config["dataset"] == "ml-latest-small":
            available_reasons_path = "reasons_cf_datasets/reasons_simsce_topp0.9_min3/reasons_simsce_topp0.9_min3.reason"
        elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
            raise NotImplementedError
        else:
            raise KeyError("The dataset is not supported yet.")
        available_reasons = pd.read_csv(available_reasons_path, sep='\t')
        available_reasons = list(available_reasons["reasons_text:token"])

    if prompt_type == "train_reason":
        train_set_reasons_num, random_selected_reasons_num = 0, 0
        reasons_file_list = os.listdir(f"reasons_cf_datasets/{config['cf_dataset_name']}")
        if "user2reasons.json" in reasons_file_list:
            reasons_file_name = "user2reasons.json"
        else:
            debias_coffecient = config["debias_coffecient"]
            reasons_file_name = f"user2reasons_{debias_coffecient}.json"
        print(f"Load reasons from {reasons_file_name}")
        
        if config["dataset"] == "ml-latest-small":
            if "7b" in config["model_name_or_path"]:
                cf_dataset_name = config["cf_dataset_name"]
                available_reasons_path = f"reasons_cf_datasets/{cf_dataset_name}/user_item2reasonsid_text.pkl"
                cf_reasons = f"reasons_cf_datasets/{cf_dataset_name}/{reasons_file_name}"
            else:
                raise NotImplementedError
        elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
            cf_dataset_name = config["cf_dataset_name"]
            available_reasons_path = f"reasons_cf_datasets/{cf_dataset_name}/user_item2reasonsid_text.pkl"
            cf_reasons = f"reasons_cf_datasets/{cf_dataset_name}/{reasons_file_name}"
        elif config["dataset"] == "Amazon_Books_small":
            cf_dataset_name = config["cf_dataset_name"]
            available_reasons_path = f"reasons_cf_datasets/{cf_dataset_name}/user_item2reasonsid_text.pkl"
            cf_reasons = f"reasons_cf_datasets/{cf_dataset_name}/{reasons_file_name}"
        else:
            raise KeyError("The dataset is not supported yet.")
        with open(available_reasons_path, 'rb') as f:
            available_reasons = pickle.load(f)
        reasonsid2text = defaultdict(list)
        for user_id, user_profile in available_reasons["user"].items():
            for reasons_id, reason_text in user_profile.items():
                reasonsid2text[reasons_id].append(reason_text[0])
        with open(cf_reasons, 'r', encoding='utf8') as f:
            user_cf2reasons = json.load(f)

    for user_id, user_feature in user_samples.items():
        his_movie = itemid2context(item_context, user_feature[2], config)
        all_candidate_list = user_feature[0]+user_feature[1]
        random.shuffle(all_candidate_list)
        candidate_movie = itemid2context(
            item_context, all_candidate_list, config)
        if config["dataset"] == "ml-latest-small":
            if prompt_type == "zero_shot":
                # prompt = f"The user has watched the following movies in chronological order: '{his_movie}'. Please recommend the next movie that the user might watch. Rank the movie recommendations from most to least recommended.\n Candidate movies: {candidate_movie}. \n The response should be formatted as: \n\"1. MovieTitle1 \n 2. MovieTitle2\n ... \""
                prompt = f"Based on the user's previous viewing of the movie '{his_movie}', please rank the candidate movies listed below. Candidate movies: {candidate_movie}. Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "few_shot":
                raise NotImplementedError
                few_shot_type = config["few_shot_type"]
                prompt = f"The user has watched the following movies: {his_movie}. Additionally, similar users have watched these movies: {similar_his}. Based on these viewing histories, please recommend a suitable movie by ranking the candidate movies listed below. The most recommended movie should be placed at the top of the list. Candidate movies: {candidate_movie}."
            elif prompt_type == "get_golden_reason":
                prompt_dict = {}
                for pos_item in user_feature[0]:
                    next_movie = itemid2context(
                        item_context, [pos_item], config)
                    # prompt = f"The user has previously watched the movie '{his_movie}' and subsequently choose to watch '{next_movie}'. Please analyze the user's viewing history and provide reasons for why they might have selected '{next_movie}' as their next movie to watch."
                    prompt = f"The user has previously watched the movie:\n'{his_movie}\nPlease analyze the user's viewing history and provide reasons for why they might have selected '{next_movie}' as their next CDs to buy."
                    prompt_dict[pos_item] = [pre_tokens + prompt + end_tokens]
                user_feature.append(prompt_dict)
            elif prompt_type == "provide_reason":
                raise NotImplementedError
                # raw_userid = model_user_id2raw_dataset_userid[user_id]
                chosen_reason = rawuserid2reasons[str(user_id)].strip()
                # prompt = f"The user has previously watched the movie: {his_movie}. The likely reason for choosing next movie was {chosen_reason}. Based on this viewing history, please recommend a suitable movie by ranking the candidate movies listed below. The most recommended movie should be placed at the top of the list. Candidate movies: {candidate_movie}."
                prompt = f"Based on the user's previous viewing of the movie '{his_movie}' and the reasons for potentially choosing to watch, listed as \n{chosen_reason}\n please rank the candidate movies listed below. Candidate movies: {candidate_movie}\n Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "random_train_reason":
                reason_numbers = config["reason_numbers"]
                chosen_reason = random.sample(
                    available_reasons, reason_numbers)
                chosen_reason = "\n".join(chosen_reason)
                prompt = f"Based on the user's previous viewing of the movie '{his_movie}' and the reasons for potentially choosing to watch, listed as \n{chosen_reason}\n please rank the candidate movies listed below. Candidate movies: {candidate_movie}\n Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "train_reason":
                user_reasons = user_cf2reasons[str(user_id)]
                user_reasons, user_reasons_score = user_reasons
                if user_reasons[0] == -1:
                    selected_reasons_list = ["No proper reasons"]
                else:
                    selected_reasons_list = []
                    tmp_available_reasons = available_reasons['user'][str(
                        user_id)]
                    for tmp_idx in range(config["train_reasons_topk"]):
                        tmp_reasons_idx = int(user_reasons[tmp_idx])
                        if "reasons_score_threshold" in config:
                            if user_reasons_score[tmp_idx] < config["reasons_score_threshold"]:
                                break
                        if tmp_reasons_idx in tmp_available_reasons.keys():
                            selected_reasons_list.append(
                                tmp_available_reasons[tmp_reasons_idx][0])
                            train_set_reasons_num += 1
                        else:
                            selected_reasons_list.append(
                                random.choice(reasonsid2text[tmp_reasons_idx]))
                            random_selected_reasons_num += 1
                    # tmp_idx = 0
                    # for _ in range(config["train_reasons_topk"]):
                    #     while tmp_idx < len(user_reasons) and user_reasons[tmp_idx] in selected_reasons_list:
                    #         tmp_idx += 1
                    #     selected_reasons_list.append(user_reasons[tmp_idx])
                    #     tmp_idx += 1
                    # selected_reasons_list = [
                    #     reasonid2text[int(i)] for i in selected_reasons_list if i != "2"]
                selected_reasons_list = "\n".join(selected_reasons_list)
                prompt = f"Based on the user's previous viewing of the movie '{his_movie}' and the reasons for potentially choosing to watch, listed as \n{selected_reasons_list}\n please rank the candidate movies listed below. Candidate movies: {candidate_movie}\n Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "zero_shot_with_reason":
                pass
            else:
                raise KeyError("The prompt_type is not supported yet.")
        elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
            if prompt_type == "zero_shot":
                # prompt = f"The user has watched the following movies in chronological order: '{his_movie}'. Please recommend the next movie that the user might watch. Rank the movie recommendations from most to least recommended.\n Candidate movies: {candidate_movie}. \n The response should be formatted as: \n\"1. MovieTitle1 \n 2. MovieTitle2\n ... \""
                prompt = f"Based on the user's previous bought CDs:\n'{his_movie}'\nPlease rank the candidate CDs listed below. Candidate CDs: {candidate_movie}. Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "get_golden_reason":
                prompt_dict = {}
                for pos_item in user_feature[0]:
                    next_movie = itemid2context(
                        item_context, [pos_item], config)
                    # prompt = f"The user has previously watched the movie '{his_movie}' and subsequently choose to watch '{next_movie}'. Please analyze the user's viewing history and provide reasons for why they might have selected '{next_movie}' as their next movie to watch."
                    prompt = f"The user has previously bought CDs:\n'{his_movie}\nPlease analyze the user's listening history and provide reasons for why they might have selected '{next_movie}' as their next CD to bought."
                    prompt_dict[pos_item] = [pre_tokens + prompt + end_tokens]
                user_feature.append(prompt_dict)
            elif prompt_type == "provide_reason":
                raise NotImplementedError
                # raw_userid = model_user_id2raw_dataset_userid[user_id]
                chosen_reason = rawuserid2reasons[str(user_id)].strip()
                # prompt = f"The user has previously watched the movie: {his_movie}. The likely reason for choosing next movie was {chosen_reason}. Based on this viewing history, please recommend a suitable movie by ranking the candidate movies listed below. The most recommended movie should be placed at the top of the list. Candidate movies: {candidate_movie}."
                prompt = f"Based on the user's previous bought CDs:\n'{his_movie}'\nand the reasons for potentially choosing to watch, listed as \n{chosen_reason}\n please rank the candidate CDs listed below. Candidate CDs: {candidate_movie}\nPlease starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "random_train_reason":
                reason_numbers = config["reason_numbers"]
                chosen_reason = random.sample(
                    available_reasons, reason_numbers)
                chosen_reason = "\n".join(chosen_reason)
                prompt = f"Based on the user's previous bought CDs:'{his_movie}' and the reasons for potentially choosing to listen, listed as \n{chosen_reason}\nPlease rank the candidate CDs listed below. Candidate CDs: {candidate_movie}\nPlease starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "train_reason":
                if user_id in model_user_id2raw_dataset_userid:
                    print(f"user_id {user_id} not in user_cf2reasons")
                    continue
                if model_user_id2raw_dataset_userid[user_id] not in user_cf2reasons:
                    print(f"user_id {user_id} not in user_cf2reasons")
                    continue
                user_reasons = user_cf2reasons[model_user_id2raw_dataset_userid[
                    user_id]]
                user_reasons, user_reasons_score = user_reasons
                if user_reasons[0] == -1:
                    selected_reasons_list = ["No proper reasons"]
                else:
                    selected_reasons_list = []
                    tmp_available_reasons = available_reasons['user'][model_user_id2raw_dataset_userid[
                        user_id]]
                    for tmp_idx in range(config["train_reasons_topk"]):
                        tmp_reasons_idx = int(user_reasons[tmp_idx])
                        if "reasons_score_threshold" in config:
                            if user_reasons_score[tmp_idx] < config["reasons_score_threshold"]:
                                break
                        if tmp_reasons_idx in tmp_available_reasons.keys():
                            selected_reasons_list.append(
                                tmp_available_reasons[tmp_reasons_idx][0])
                            train_set_reasons_num += 1
                        else:
                            selected_reasons_list.append(
                                random.choice(reasonsid2text[tmp_reasons_idx]))
                            random_selected_reasons_num += 1
                selected_reasons_list = "\n".join(selected_reasons_list)
                prompt = f"Based on the user's previous bought CDs:'{his_movie}' and the reasons for potentially choosing to listen, listed as \n{selected_reasons_list}\nPlease rank the candidate CDs listed below. Candidate CDs: {candidate_movie}\nPlease starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            else:
                raise KeyError("The prompt_type is not supported yet.")
        elif config["dataset"] == "Amazon_Books_small":
            if prompt_type == "zero_shot":
                prompt = f"Based on the user's previous bought books:\n'{his_movie}'\nPlease rank the candidate books listed below. Candidate books: {candidate_movie}. Please starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            elif prompt_type == "get_golden_reason":
                prompt_dict = {}
                for pos_item in user_feature[0]:
                    next_movie = itemid2context(
                        item_context, [pos_item], config)
                    prompt = f"The user has previously bought books:\n'{his_movie}\nPlease analyze the user's reading history and provide reasons for why they might have selected '{next_movie}' as their next book to bought."
                    prompt_dict[pos_item] = [pre_tokens + prompt + end_tokens]
                user_feature.append(prompt_dict)
            elif prompt_type == "train_reason":
                # if user_id not in model_user_id2raw_dataset_userid:
                #     print(f"user_id {user_id} not in user_cf2reasons")
                #     continue
                # if model_user_id2raw_dataset_userid[user_id] not in user_cf2reasons:
                #     print(f"raw_user_id {model_user_id2raw_dataset_userid[user_id]} not in user_cf2reasons")
                #     continue
                # user_reasons = user_cf2reasons[model_user_id2raw_dataset_userid[
                #     user_id]]
                user_reasons = user_cf2reasons[user_id]
                user_reasons, user_reasons_score = user_reasons
                if user_reasons[0] == -1:
                    selected_reasons_list = ["No proper reasons"]
                else:
                    selected_reasons_list = []
                    tmp_available_reasons = available_reasons['user'][user_id]
                    for tmp_idx in range(config["train_reasons_topk"]):
                        tmp_reasons_idx = int(user_reasons[tmp_idx])
                        if "reasons_score_threshold" in config:
                            if user_reasons_score[tmp_idx] < config["reasons_score_threshold"]:
                                break
                        if tmp_reasons_idx in tmp_available_reasons.keys():
                            selected_reasons_list.append(
                                tmp_available_reasons[tmp_reasons_idx][0])
                            train_set_reasons_num += 1
                        else:
                            selected_reasons_list.append(
                                random.choice(reasonsid2text[tmp_reasons_idx]))
                            random_selected_reasons_num += 1
                selected_reasons_list = "\n".join(selected_reasons_list)
                prompt = f"Based on the user's previous bought books:'{his_movie}' and the reasons for potentially choosing to read, listed as \n{selected_reasons_list}\nPlease rank the candidate books listed below. Candidate books: {candidate_movie}\nPlease starting with the most recommended."
                user_feature.append(pre_tokens + prompt + end_tokens)
            else:
                raise KeyError("The prompt_type is not supported yet.")
        else:
            raise KeyError("The dataset is not supported yet.")
    if prompt_type == "train_reason":
        logger.info(
            f"trainset_reasons{train_set_reasons_num}_random_reasons{random_selected_reasons_num}")
    return user_samples


if __name__ == '__main__':

    # configurations initialization
    print("Data Loading for RecBole...")

    config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
        "config/sequential_ml.yaml", "config/LLM.yaml"])

    init_logger(config)
    init_seed(config['seed'], config['reproducibility'])
    logger = getLogger()
    train_data, valid_data, test_data = load_recbole_datasets(logger, config)
    user_context, item_context, user_item_inter = load_context_data(
        logger, config)

    user_dict = seqdataloader_convert2prompt(
        user_context, item_context, test_data, config["prompt_type"], logger, config, test_data._dataset.field2token_id["item_id"], test_data._dataset.field2token_id["user_id"])

    with open("get_reasons_with_openai.json", 'w', encoding='utf8') as f:
        json.dump(user_dict, f)
    # for item in train_data:
    #     continue
    # for item in test_data:  # cur_data, idx_list, positive_u, positive_i
    #     continue
    # print("Down")
