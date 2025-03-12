export CUDA_VISIBLE_DEVICES=2
python recommendation_prompt_generation.py --dataset ml-latest-small --topk_reasons 5 --debias_coffecient 5
python recommendation_prompt_generation.py --dataset Amazon_CDs_and_Vinyl_small --topk_reasons 1 --debias_coffecient 1
python recommendation_prompt_generation.py --dataset Amazon_Books_small --topk_reasons 10 --debias_coffecient 1

# Rank with LLMs
python recommendation_prompt4LLM/openai_recommendation.py

# Convert format for evaluation
python recommendation_prompt4LLM/combinations.py

# Evaluate
python llm_evaluation.py


