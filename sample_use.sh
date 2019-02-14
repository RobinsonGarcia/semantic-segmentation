#!bash/bin

python SemanticSegment/build_random_search_params.py

python SemanticSegment/experiments.py --model_dir Shiriu --number_experiments 80 --random_params random_search_params.pickle

