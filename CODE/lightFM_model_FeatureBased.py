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

def get_similar_tags(model, tag_id):
    # Define similarity as the cosine of the angle
    # between the tag latent vectors

    # Normalize the vectors to unit length
    tag_embeddings = (model.item_embeddings[18090:,:]
                      / np.linalg.norm(model.item_embeddings[18090:,:], axis=1).reshape((-1,1)))

    query_embedding = tag_embeddings[tag_id]
    similarity = np.dot(tag_embeddings, query_embedding)
    most_similar = np.argsort(-similarity)[1:4]

    return most_similar

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

# add features
all_categories = []
for categories in businessDF['categories'].tolist():
  if categories == None:
    continue
  for category in categories.split(','):
    all_categories.append(category.rstrip().lstrip())
all_categories = set(all_categories)
all_categories = list(all_categories)
print(len(all_categories))
reviewDataset.fit_partial(items=(GA_restaurant_dict[bid] for bid in businessDF['business_id'].tolist()), item_features=all_categories)
num_items, num_features = reviewDataset.item_features_shape()
print("Feature shape", num_items, num_features)
# businessIDList = businessDF['review_count'].tolist()
# businessIDSet = set(businessIDList)
# reviewDataset.fit_partial(items=(GA_restaurant_dict[bid] for bid in businessDF['business_id'].tolist()), item_features=(review_count for review_count in businessDF['review_count'].tolist()))
# num_items, num_features = reviewDataset.item_features_shape()
# print("Feature shape", num_items, num_features)

# check length of user and item index


(interactions, weights) = reviewDataset.build_interactions(((data.loc['user_id'], data.loc['business_id']) for (_, data) in reviewDF.iterrows()))
print(repr(interactions))
# build restaurant features
# item_features = reviewDataset.build_item_features(((GA_restaurant_dict[row['business_id']], [row['review_count'], row['categories']]) for (_, row) in businessDF.iterrows()))\
category_list = []
for categories in businessDF['categories'].tolist():
  if categories == None:
    category_list.append([])
    continue
  local_catgories = []
  for category in categories.split(','):
    local_catgories.append(category.lstrip().rstrip())
  category_list.append(local_catgories)
item_features = reviewDataset.build_item_features(((GA_restaurant_dict[row['business_id']], category_list[i]) for i, (_, row) in enumerate(businessDF.iterrows())))
print(repr(item_features))

trainset, testset = random_train_test_split(interactions, test_percentage=.1)
model = LightFM(no_components=100, loss='warp')
model.fit(trainset, item_features=item_features, epochs=5, verbose=True)

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
print("AUC score: %.2f" % auc_score(model, testset, item_features=item_features).mean() )
# print("Train precision: %.2f" % precision_at_k(model, trainset, k=10).mean())
# print("Test precision: %.2f" % precision_at_k(model, testset, k=10).mean())


all_categories = np.array(all_categories)
for tag in (u'American (Traditional)', u'Japanese', u'Cafes'):
    
    tag_id = all_categories.tolist().index(tag)
    print('Most similar tags for %s: %s' % (all_categories[tag_id],
                                            all_categories[get_similar_tags(model, tag_id)]))


# Evaluation
# surprise: rating prediction -> RMSE
# lightFM: 1. 没有rating过的 -> negative 2. rating过的 ——> positive
# interpretability: SVD: embedding , lightFM: category
# AUC
# RMSE， 