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
from surprise import BaselineOnly

original_stdout = sys.stdout
with open('baseline_model_result.txt', 'w') as f:
  sys.stdout = f
# 504886 reviews in GA.csv


  reviewDF = pd.read_csv("GA.csv")
  reader = Reader(rating_scale=(1, 5))
  totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
  trainset, testset = train_test_split(totalDataset, test_size=.1)

  # We'll use the famous SVD algorithm.
  print('Using SGD')
  bsl_options = {'method': 'sgd',
                'learning_rate': .00005,
                }
  algo = BaselineOnly(bsl_options=bsl_options)

  # Train the algorithm on the trainset, and predict ratings for the testset
  algo.fit(trainset)
  predictions = algo.test(testset)

  # Then compute RMSE
  accuracy.rmse(predictions)

sys.stdout = original_stdout


