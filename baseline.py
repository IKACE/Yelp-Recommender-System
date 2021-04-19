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
ratings = reviewDF['stars'].to_numpy()
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
for key in userIDCount:
    ratings = userIDCount[key]
    userIDCount[key] = sum(ratings) / len(ratings)
mu = np.mean(ratings)
original_stdout = sys.stdout
with open("baseline_results.txt", 'w') as f:
    sys.stdout = f
    print("mean ratings in GA ", mu)
    print("==== businessID to avg rating ===")
    print(businessIDCount)
    print("==== userIDCount to avg rating ===")
    print(userIDCount)
sys.stdout = original_stdout