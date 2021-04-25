"""Code for data processing and plotting of dataset"""

import json
import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# matplotlib.use('TkAgg')
business_path = "./dataset/yelp_academic_dataset_business.json"
review_path = "./dataset/yelp_academic_dataset_review.json"
user_path = "./dataset/yelp_academic_dataset_user.json"

def create_dataset(line):
    dataset = {}
    keys = line.keys()
    for k in keys:
        dataset[k]= []
    return dataset, keys

def parse_json(filename):
    dataset = {}
    keys = []
    with open(filename, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for count, line in enumerate(lines):
            data = json.loads(line.strip())
            if count == 0:
                dataset, keys = create_dataset(data)
            for k in keys:
                dataset[k].append(data[k])                
        return pd.DataFrame(dataset)



# reviews_per_state()
# reviews_per_business()
# businessDF = parse_json(business_path)
# reviewDF = parse_json(review_path)
# stateList = businessDF['state'].to_numpy()
# businessIDList = businessDF['business_id'].to_numpy()
# businessIDList = businessIDList[np.where(stateList == 'GA')]
# businessIDList = businessIDList.tolist()
# ratings = reviewDF[reviewDF['business_id'].isin(businessIDList)]['stars'].to_numpy()
# mu = np.mean(ratings)

def GA_processing():
        # # business in GA
    # businessIDs = []
    # # users rating >= 10 reviews in GA
    # userIDs = []
    # reviewDF = parse_json(review_path)
    # reviewDF = reviewDF[reviewDF['business_id'].isin(businessIDs)]
    # reviewDF = reviewDF[reviewDF['user_id'].isin(userIDs)]
    # print("original df length", len(reviewDF.index))
    # # sample for a test set
    # reviewDFTest = reviewDF.sample(frac=0.1)
    # print("test df length", len(reviewDFTest.index))
    # testBusinessIDs = reviewDFTest['business_id'].tolist()
    # reviewDF = reviewDF[~reviewDF['business_id'].isin(testBusinessIDs)]
    # # sample for a val set
    # reviewDFVal = reviewDF.sample(frac=0.1)
    # print("val df length", len(reviewDFVal.index))
    # valBusinessIDs = reviewDFVal['business_id'].tolist()
    # reviewDF = reviewDF[~reviewDF['business_id'].isin(valBusinessIDs)]

    # print("final train df length", len(reviewDF.index))
    GA_restaurant_dict = {}
    GA_user_dict = {}

    with open('GA_restaurants_indices.json') as f:
        GA_restaurant_dict = json.load(f)

    with open('GA_users_indices.json') as f:
        GA_user_dict = json.load(f)

    GA_restaurant_IDs = list(GA_restaurant_dict)
    GA_user_IDs = list(GA_user_dict)

    reviewDF = parse_json(review_path)
    print("=== Parsing complete! ===")
    reviewDF = reviewDF[reviewDF['business_id'].isin(GA_restaurant_IDs)]
    reviewDF = reviewDF[reviewDF['user_id'].isin(GA_user_IDs)]
    print("=== Filtering complete! ===")
    reviewDF['user_id'] = reviewDF['user_id'].map(GA_user_dict)
    reviewDF['business_id'] = reviewDF['business_id'].map(GA_restaurant_dict)
    # reviewDF.replace({'user_id': GA_user_dict}, inplace=True)
    # reviewDF.replace({'business_id': GA_restaurant_dict}, inplace=True)
    print("=== Replacing complete! ===")
    reviewDF.to_csv("GA.csv")    

# businessDF = parse_json(business_path)
# businessDF.to_csv('business.csv')

def generate_category_index():
    businessDF = parse_json(business_path)
    categoryList = businessDF['categories'].tolist()
    print(categoryList[0].split(','))
    # print(len(reviewCountList), len(reviewCountSet))

# generate_category_index()
# reviews_per_state()
# reviews_per_business()
# reviews_per_user()