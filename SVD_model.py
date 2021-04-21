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
from surprise import SVD

original_stdout = sys.stdout
with open('SVD_model_result.txt', 'w') as f:
    sys.stdout = f
    reviewDF = pd.read_csv("GA.csv")
    reader = Reader(rating_scale=(1, 5))
    totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
    # trainset, testset = train_test_split(totalDataset, test_size=.1)
    print("Finish loading data")
    candidateMeasures = {}
    paramChoices = {
        'factor': [50, 100, 150],
        'epoch': [5],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.2, 0.4, 0.8] 
    }
    for f in paramChoices['factor']:
        for n in paramChoices['epoch']:
            for l in paramChoices['lr_all']:
                for r in paramChoices['reg_all']:
                    algo = SVD(n_factors = f, n_epochs = n, lr_all = l, reg_all = r)
                    print("===== evaluating parameter choice", f, n, l, r)
                    candidateMeasures[f,n,l,r] = cross_validate(algo, totalDataset, measures=['RMSE', 'MAE'], cv=10, verbose=True)

    print(candidateMeasures)
    # RMSEs = dict(sorted(RMSEs.items(), key=lambda item: item[1]))

    # for key in RMSEs:
    #     print(key, RMSEs[key])
    #     break

sys.stdout = original_stdout


