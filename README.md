# CausalX

# Quick Verification

We have provided the final results and all intermediate outcomes in json or pkl formats for verification. To save storage space, all results have been compressed into ZIP format.

- **Step 1: Candidate Explanations Generation**

  - Access raw generated explanations by OpenAI at `candidate_reasons/openai_reasons_cache/`.
  - Constructed database is located at `reasons_cf_datasets/`.
- **Step 2: Debias Explanation Selection**

  - Selected explanations are available at `reasons_cf_datasets/dataset_name/user2reasons.json`.
- **Step 3: Recommendation Prompt and Results**

  - Recommendation prompt is provided at `recommendation_prompt4LLM/with_reasons_outputs`.
  - Recommendation results can be found at `recommendation_prompt4LLM/with_reasons_results`.

# Getting Started

Please note, the complete implementation code is not included here. Necessary modifications to the Python RecBole package (sections with ~~strikethrough~~ below) are also omitted. The full code will be released after our paper is published.

## Required Packages

```markdown
cuml==24.4.0
torch==2.3.0
recbole==1.2.0
scipy==1.12.0
scikit-learn==1.4.1
sentence-transformers==2.7.0
transformers==4.40.2
vllm==0.4.2
```

## Modifications of Recbole Package

Please copy the scripts from the `recbole/` directory to your Python RecBole package directory to replace the existing files.

```markdown
recbole/
├── trainer/
│   └── trainer.py
├── data/
│   ├── dataloader/
│   │   └── general_dataloader.py
│   └── utils.py
├── sampler/
    └── sampler.py
```

## Datasets

We perform our experiments on the [Movielens](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip), [Amazon_CDs_and_Vinyl](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz) and [Amazon_Books](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books_5.json.gz) datasets. Data is converted to the RecBole-specific format using their official script, available at [RecBole Datasetes](https://github.com/RUCAIBox/RecSysDatasets). Processed data and our subset creation scripts are located in the `data/` folder.

### Note for Dataset Usage:

This setup is designed for the MovieLens datasets. Please For the Amazon_CDs_and_Vinyl dataset, change the dataset variable to `amazoncd` at the beginning of each script for Amazon_CDs_and_Vinyl dataset and `amazonbook` for Amazon Books dataset. Check paths to ensure they are absolute to avoid bugs related to relative addressing.

## Detailed Steps

### Step 1: Candidate Explanations Generation

Generate candidate explanations using the OpenAI API:

```bash

# get the prompts for explanations generation
python get_rs_reasons_prompt.py

# generate explanations with API
# Please note here it is also available to use openai batchAPI for less cost, which can reference candidate_reasons/openai_batch.ipynb
python candidate_reasons/openai.py

# Combine the prompt with result
python candidate_reasons/correct_id.py

# construct the database of candidate explanations
python candidate_reasons/dbscan_embedding_openai.py
```

or you can direct run `bash script/explanation_generation.sh`

Please note here it is also available to use openai batchAPI for less cost, which can reference `candidate_reasons/openai_batch.ipynb`

### Step 2: Debias Explanation Selection

```bash

# Calculate item popularity
python calculate_popularity.py

# Train the debias model
bash rs_models/scripts/CF_model_train_movielens_pop.sh
bash rs_models/scripts/CF_model_train_amazoncd_pop.sh
bash rs_models/scripts/CF_model_train_amazonbook_pop.sh

# Retrieve selected explanations
python CF_model_test_debias.py --dataset ml-latest-small --debias_coeffiecient 5 --best_model_path your_model_save_path
python CF_model_test_debias.py --dataset Amazon_CDs_and_Vinyl_small --debias_coeffiecient 1 --best_model_path your_model_save_path
python CF_model_test_debias.py --dataset Amazon_Books_small --debias_coeffiecient 1 --best_model_path your_model_save_path
```

### Step 3: LLM-based Recommendation

```bash
# Generate recommendation prompts
python recommendation_prompt_generation.py --dataset ml-latest-small --topk_reasons 5 --debias_coffecient 5
python recommendation_prompt_generation.py --dataset Amazon_CDs_and_Vinyl_small --topk_reasons 1 --debias_coffecient 1
python recommendation_prompt_generation.py --dataset Amazon_Books_small --topk_reasons 10 --debias_coffecient 1

# Rank with LLMs
python recommendation_prompt4LLM/openai_recommendation.py

# Convert format for evaluation
python recommendation_prompt4LLM/combinations.py

# Evaluate
python llm_evaluation.py
```

or you can just run ` bash script/LLM_recommendation.sh`

## Baselines and Local LLM Usage

We also provide implementation details for some baseline models and guidance on using local LLMs with adjusted hyperparameters for proper output generation.

**Note:** The setup details for the baseline models have not yet been fully validated. Hyperparameter adjustments may be required, and some bugs may still need to be resolved.

```bash

# Step1 
# get the prompts for explanations generation 
python get_rs_reasons.py 
python correct_id.py 
# construct the database of candidate explanations 
python candidate_reasons/dbscan_embedding.py 


# Step2 is same
# Calculate item popularity
python calculate_popularity.py
# Train the debias model
bash rs_models/scripts/CF_model_train_movielens_pop.sh
bash rs_models/scripts/CF_model_train_amazoncd_pop.sh
# Retrieve selected explanations
python CF_model_test_debias.py


# Step 3 
python LLMRS_prompt.py
```

# Reference

```
@article{li2025counter,
      title={Counterfactual Language Reasoning for Explainable Recommendation Systems}, 
      author={Guanrong Li and Haolin Yang and Xinyu Liu and Zhen Wu and Xinyu Dai},
      journal={arXiv preprint arXiv:2503.08051},
      year={2025},
      url={https://arxiv.org/abs/2503.08051}, 
}
```
