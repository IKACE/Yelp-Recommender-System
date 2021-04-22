import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from collections import Counter
from dataprocessing import parse_json, business_path, review_path, user_path
from lightfm.evaluation import auc_score, precision_at_k


from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split

# TODO: IDEA: transform category into a one hot feature form

with open('GA_restaurants_indices.json') as f:
  GA_restaurant_dict = json.load(f)
reviewDF = pd.read_csv("GA.csv")
businessDF = parse_json(business_path)
businessDF = businessDF[businessDF['business_id'].isin(GA_restaurant_dict)]
reviewDataset = Dataset()
# build inner index of dataset
reviewDataset.fit((user_id for user_id in reviewDF['user_id'].tolist()), (business_id for business_id in reviewDF['business_id'].tolist()))
num_users, num_items = reviewDataset.interactions_shape()
print(num_users, num_items)


# check length of user and item index


(interactions, weights) = reviewDataset.build_interactions(((data.loc['user_id'], data.loc['business_id']) for (_, data) in reviewDF.iterrows()))
print(repr(interactions))

trainset, testset = random_train_test_split(interactions, test_percentage=.1)
model = LightFM(no_components=100, loss='warp')
model.fit(trainset, epochs=10, verbose=True)

# # prediction 
# userIDList = reviewDF['user_id'].tolist()
# businessIDList = reviewDF['business_id'].tolist()
# userCount = len(userIDList)
# randomList = random.sample(range(0,userCount), 50)
# for i, index in enumerate(randomList):
#   randomList[i] = userIDList[index]
# num_users, num_items = reviewDataset.interactions_shape()
# for userID in randomList:
#     prediction = model.predict(userID, businessIDList)
#     prediction.sort(reverse=True)
#     print(prediction)
# prediction = model.predict()
print("AUC score: %.2f" % auc_score(model, testset).mean() )
# print("Train precision: %.2f" % precision_at_k(model, trainset, k=10).mean())
# print("Test precision: %.2f" % precision_at_k(model, testset, k=10).mean())