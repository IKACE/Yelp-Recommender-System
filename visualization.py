import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
from dataprocessing import parse_json, business_path, review_path, user_path
from sklearn.manifold import TSNE
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import SVD
# 150 5 0.01 0.2
businessDF = parse_json(business_path)
with open('GA_restaurants_indices.json') as f:
  GA_restaurant_dict = json.load(f)
businessDF = businessDF[businessDF['business_id'].isin(GA_restaurant_dict)]
reviewDF = pd.read_csv("GA.csv")
reader = Reader(rating_scale=(1, 5))
totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
trainset, testset = train_test_split(totalDataset, test_size=.1)
algo = SVD(n_factors = 150, n_epochs = 5, lr_all = 0.01, reg_all = 0.02)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
tsne = TSNE(n_components=2, n_iter=500, verbose=3, random_state=6)
businessEmbedding = tsne.fit_transform(algo.qi)
embeddingDF = pd.DataFrame(columns=['x', 'y'], data=businessEmbedding)
embeddingDF.plot.scatter('x', 'y')
plt.show()