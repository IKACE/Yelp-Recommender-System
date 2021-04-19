"""Code for baseline linear model"""
import json
import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter
from dataprocessing import parse_json, business_path, review_path, user_path
businessDF = parse_json(business_path)
reviewDF = parse_json(review_path)
stateList = businessDF['state'].to_numpy()
businessIDList = businessDF['business_id'].to_numpy()
businessIDList = businessIDList[np.where(stateList == 'GA')]
businessIDList = businessIDList.tolist()
reviewDF = reviewDF[reviewDF['business_id'].isin(businessIDList)]
allRatings = reviewDF['stars'].to_numpy()
# userID -> ratings, businessID -> ratings
userIDCount = {}
businessIDCount = {}
for index in reviewDF.index:
    rating = reviewDF['stars'][index]
    businessID = reviewDF['business_id'][index]
    userID = reviewDF['user_id'][index]
    if businessID not in businessIDCount:
        businessIDCount[businessID] = []
    businessIDCount[businessID].append(rating)
    if userID not in userIDCount:
        userIDCount[userID] = []
    userIDCount[userID].append(rating)
for key in businessIDCount:
    ratings = businessIDCount[key]
    businessIDCount[key] = sum(ratings) / len(ratings)
for key in list(userIDCount):
    ratings = userIDCount[key]
    if len(ratings) < 10:
        userIDCount.pop(key, None)
    else:
        userIDCount[key] = sum(ratings) / len(ratings)
mu = np.mean(allRatings)
original_stdout = sys.stdout
with open("baseline_results_new.txt", 'w') as f:
    sys.stdout = f
    print("mean ratings in GA ", mu)
    print("==== businessID to avg rating ===")
    print(businessIDCount)
    print("==== userIDCount to avg rating ===")
    print(userIDCount)
    print("==== userID that has more than or equal to 10 reviews ====")
    print(userIDCount.keys())
    print("==== all business id in GA ====")
    print(businessIDCount.keys())
sys.stdout = original_stdout