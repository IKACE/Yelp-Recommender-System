import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
from dataprocessing import parse_json, business_path, review_path, user_path

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import NMF

original_stdout = sys.stdout
# ===== evaluating parameter choice 10 30 0.08 0.08 True
with open('NMF_model_result.txt', 'w') as f:
    sys.stdout = f
    reviewDF = pd.read_csv("GA.csv")
    reader = Reader(rating_scale=(1, 5))
    totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
    # trainset, testset = train_test_split(totalDataset, test_size=.1)
    print("Finish loading data")
    candidateMeasures = {}
    paramChoices = {
        'factor': [30, 45],
        'epoch': [50, 70],
        'reg_qi': [0.06, 0.08, 0.1, 0.2],
        'reg_pu': [0.06, 0.08, 0.1, 0.2],
        'biased': [True]
    }
    for f in paramChoices['factor']:
        for n in paramChoices['epoch']:
            for rqi in paramChoices['reg_qi']:
                for rpu in paramChoices['reg_pu']:
                    for biased in paramChoices['biased']:
                        algo = NMF(n_factors = f, n_epochs = n, reg_qi = rqi, reg_pu = rpu, biased = biased)
                        print("===== evaluating parameter choice", f, n, rqi, rpu, biased)
                        candidateMeasures[f,n,rqi,rpu, biased] = cross_validate(algo, totalDataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    print(candidateMeasures)
    # RMSEs = dict(sorted(RMSEs.items(), key=lambda item: item[1]))

    # for key in RMSEs:
    #     print(key, RMSEs[key])
    #     break

sys.stdout = original_stdout


