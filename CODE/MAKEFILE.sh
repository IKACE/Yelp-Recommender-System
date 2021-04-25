#! /bin/bash

# we recommend you downloading our GA.csv and .json files as specified in README.md
# in that case, comment line 8 to line 12 of this MAKEFILE.sh

pip3 install numpy pandas matplotlib surprise lightfm sklearn plotly datapane &&

python3 filter_GA_restaurants_and_users.py &&

python3 GA_resturants_and_users_to_index.py && 

python3 dataprocessing.py && 

python3 baseline_model.py && 

python3 SVD_model.py &&

python3 NMF_model.py &&

python3 lighFM_model_FeatureBased.py &&

python3 prediction_recall_k_solver.py &&

python3 visualization.py