import pandas as pd
import json
import re
import torch
import wandb


class VirtueLLM:
    def parameters(self):
        return [torch.Tensor([1., 2., 3.])]


def extract_titles(text):
    line = text.strip()
        
    # 处理带sells for格式的行
    if 'sells for' in line:
        title = line.split("', sells for")[0].strip("'")
        
    # 处理带价格符号$的行    
    elif ' - $' in line:
        title = line.split(' - $')[0].strip('"').strip("'")
        
    # 处理纯标题格式(被引号包围)
    elif line.startswith('"') or line.startswith("'"):
        title = line.strip('"').strip("'")
    else:
        title = line

    return title

def load_rank(movie_title2id: dict, llm_output: str, config):
    ranked_results = []

    # Process only for specific configurations
    if "openai" in config["exp_name"]:
        # Split LLM output into lines and clean up titles
        tmp_results = llm_output.split("\n")
        tmp_results = [re.sub(r'^\d+\.\s*', '', i).strip() for i in tmp_results]
        tmp_results = [re.sub(r"^\s*['\"](.*?)['\"]\s*$", r'\1', i) for i in tmp_results]

        # Match and validate each title
        for i in tmp_results:
            if i not in movie_title2id:
                # Try extracting titles from different patterns
                patterns = [
                    r"'(.*)'",  # Single quotes
                    r'"(.*?)"',  # Double quotes
                    r"\"(.*)\"|'(.*)'"  # Either double or single quotes
                ]

                for pattern in patterns:
                    match = re.search(pattern, i)
                    if match:
                        tmp_i = match.group(1) or match.group(2)
                        if tmp_i in movie_title2id:
                            i = tmp_i
                            break
                else:
                    # Attempt title extraction with custom logic
                    tmp_i = extract_titles(i)
                    if tmp_i in movie_title2id:
                        i = tmp_i
                    else:
                        continue

            # Add to results if valid
            if i in movie_title2id:
                ranked_results.append(movie_title2id[i])
            else:
                print(i)

        # Log the number of ranked results
        # print(len(ranked_results))

    return ranked_results

    if "llama2" in config["exp_name"] or "llama3" in config["exp_name"] or "mistral" in config["exp_name"]:
        llm_output = llm_output.split("most recommended movies are:")
        if len(llm_output) == 1:
            llm_output = llm_output[0].split("most recommended are:")
            if len(llm_output) == 1:
                llm_output = llm_output[0]
            else:
                llm_output = llm_output[1]
        else:
            llm_output = llm_output[1]
        if config["dataset"] == "ml-latest-small":
            pattern = re.compile(
                r'(\d+)\.["|\']?([^"]+?)["|\']?\s+\((\d{4})\)')
            matches = pattern.findall(llm_output)
            ranked_movies = [title for idx, title, year in matches]
            if len(matches) == 0:
                pattern = re.compile(r'[\'|"]([^"]+?)[\'|"]\s+\((\d{4})\)')
                matches = pattern.findall(llm_output)
                ranked_movies = [title for title, year in matches]
            if len(matches) == 0:
                pattern = re.compile(r'\n(\d+)\.\s*["|\']([^"]+?)["|\']')
                matches = pattern.findall(llm_output)
                ranked_movies = [title for idx, title in matches]
            if len(matches) == 0:
                pattern = re.compile(r'["|\']([^"]+)["|\'],')
                matches = pattern.findall(llm_output)
                ranked_movies = [title for title in matches]
            if len(matches) == 0:
                pattern = re.compile(r'\n(\d+)\.\s+(.+?)\s+\(.*?\d+?\)')
                matches = pattern.findall(llm_output)
                ranked_movies = [title for idx, title in matches]
        elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
            ranked_movies = []
            pattern = re.compile(r'(\d+)\.\s(.*?)-.*?\(\d{4}\)\s\-')
            matches = pattern.findall(llm_output)
            tmp_ranked_movies = [title.strip() for idx, title
                                 in matches if title.strip() in movie_title2id]
            if len(tmp_ranked_movies) >= len(ranked_movies):
                ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
            if len(tmp_ranked_movies_no) >= len(ranked_movies):
                ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s[\'|\"]?(.*?)[\'|\"]?\s\(\d{4}\)\s\-')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(r'(\d+)\.\s[\'|\"]?(.*?)[\'|\"]?\s-')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s["|\']?(.*?)["|\']?\sby\s(.*?)\s-')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title, tmp
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title, tmp
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s[\'|\"](.*?)[\'|\"]')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                    ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s[\'|\"]?(.*?)[\'|\"]?-')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s(.*?)\s\(.*?\)\s-')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s(.*?):')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s(.*?)\sby\s.*?:')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) < 3:
                pattern = re.compile(
                    r'(\d+)\.\s[\'|"]??(.*?)[\'|"]??,\swhich.*?')
                matches = pattern.findall(llm_output)
                tmp_ranked_movies = [title.strip() for idx, title,
                                     in matches if title.strip() in movie_title2id]
                if len(tmp_ranked_movies) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies
                tmp_ranked_movies_no = [title.strip().strip('\'"') for idx, title,
                                        in matches if title.strip().strip('\'"') in movie_title2id]
                if len(tmp_ranked_movies_no) >= len(ranked_movies):
                    ranked_movies = tmp_ranked_movies_no
            if len(ranked_movies) == 0:
                pass
        else:
            raise NotImplementedError("Not implemented dataset")
        # Extracting movies and displaying them
        # if len(ranked_movies) == 0:
        #     print(llm_output)
        for per_movie in ranked_movies:
            per_movie = per_movie.strip()
            if per_movie[0] == "'" and per_movie[-1] == "'":
                per_movie = per_movie[1:-1]
            if per_movie[-1] == "\"" and per_movie[-1] == "\"":
                per_movie = per_movie[1:-1]
            if per_movie[0] == "\'" or per_movie[0] == "\"":
                per_movie = per_movie[1:]
            if per_movie not in movie_title2id:
                # print(per_movie)
                continue
            ranked_results.append(movie_title2id[per_movie])
    else:
        raise NotImplementedError("Not implemented model")
    if len(ranked_results) == 0:
        pass
    return ranked_results


def load_results(result_dict, saved_file_path, item_features, config, logger, raw_dataset_id2model_item_id):
    if result_dict is None:
        result_dict = json.load(open(saved_file_path, 'r', encoding='utf8'))
    movie_title2id = {}
    for item_id, item_feature in item_features.items():
        if config["dataset"] == "ml-latest-small":
            title = item_feature['movie_title:token_seq']
        elif config["dataset"] == "Amazon_CDs_and_Vinyl_small":
            title = item_feature["title:token"]
        elif config["dataset"] == "Amazon_Books_small":
            title = item_feature["title:token"]
        else:
            raise NotImplementedError("Not implemented dataset")
        item_id = str(item_id)
        if item_id not in raw_dataset_id2model_item_id:
            continue
        movie_title2id[title] = raw_dataset_id2model_item_id[str(item_id)]

    user_idx = 0
    pred_u, pred_i, pred_value, pos_u, pos_i = [], [], [], [], []
    not_compile_number = 0
    for key, value in result_dict.items():
        if len(value) == 5:
            added_pred_i = load_rank(movie_title2id, value[4], config)
        else:
            # print(value)
            continue
        if len(added_pred_i) == 0:
            not_compile_number += 1
            continue
        # pred_i += added_pred_i
        # pred_u += [user_idx] * len(added_pred_i)
        # pred_value += list(range(len(added_pred_i), 0, -1))
        user_pos_model_id = [raw_dataset_id2model_item_id[raw_idx]
                             for raw_idx in value[0]]
        user_neg_model_id = [raw_dataset_id2model_item_id[raw_idx]
                             for raw_idx in value[1]]
        user_all_eval_id = user_pos_model_id + user_neg_model_id
        tmp_added_pred_i = []
        for per_pred_i in added_pred_i:
            if per_pred_i in user_all_eval_id:
                tmp_added_pred_i.append(per_pred_i)
        for per_eval_i in user_all_eval_id[::-1]:
            if per_eval_i not in tmp_added_pred_i:
                tmp_added_pred_i.append(per_eval_i)
        pred_i += tmp_added_pred_i
        pred_u += [user_idx] * len(tmp_added_pred_i)
        pred_value += list(range(len(tmp_added_pred_i), 0, -1))

        # for pos_item_value in value[0]:
        #     if pos_item_value not in pred_i:
        #         pred_i.append(pos_item_value)
        #         pred_u.append(user_idx)
        #         pred_value.append(0)

        pos_i += [raw_dataset_id2model_item_id[raw_idx]
                  for raw_idx in value[0]]
        pos_u += [user_idx] * len(value[0])
        user_idx += 1

        assert len(pred_u) == len(pred_i)
        assert len(pred_u) == len(pred_value)
        assert len(pos_u) == len(pos_i)
    logger.info(f"Not compile number for LLM: {not_compile_number}")
    return pred_u, pred_i, pred_value, pos_u, pos_i


if __name__ == "__main__":
    # configurations initialization
    from logging import getLogger
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    from recbole.trainer import Trainer

    from load_datasets import load_recbole_datasets, load_context_data, seqdataloader_convert2prompt

    from vllm import LLM
    from vllm import SamplingParams
    
    from time import sleep
    
    import os
    print("Data Loading for RecBole...")
    
    
    dataset = "amazonbooks"
    saved_dirs = "recommendation_prompt4LLM/with_reasons_combinations/"
    saved_file = os.listdir(saved_dirs)
    saved_file = [saved_file[21]]
    for per_file in saved_file:
        if dataset not in per_file:
            continue
        saved_path = saved_dirs + per_file
    
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
            raise NotImplementedError("Not implemented dataset")
        
        config["exp_name"] = saved_path.split("/")[-1].replace(".json", "")
        config["special_key"] = saved_path.split("/")[-1].replace(".json", "")
        print(config["exp_name"])
        init_logger(config)
        init_seed(config['seed'], config['reproducibility'])
        logger = getLogger()
        logger.info(config)

        train_data, valid_data, test_data = load_recbole_datasets(logger, config)
        user_context, item_context, user_item_inter = load_context_data(
            logger, config)

        setattr(config, "recommend_saved_path", saved_path)

        # 检查是否存在预期的键和对应的值

        pred_u, pred_i, value, pos_u, pos_i = load_results(
            None, saved_path, item_context, config, logger, test_data._dataset.field2token_id["item_id"])

        config["device"] = "cpu"
        trainer = Trainer(config, VirtueLLM())
        trainer.evaluate([pred_u, pred_i, value, pos_u, pos_i,
                        test_data._dataset.item_num], None)
        print(config["exp_name"])
        print("--------------------------------")
        sleep(10)
