import json
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from dataprocessing import parse_json, business_path, review_path, user_path
from sklearn.manifold import TSNE
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise import SVD
import plotly.express as px
import datapane as dp

# TODO: for plot 1, color restaurants by categories

# old 150 5 0.01 0.2
# new 100 15 0.02 0.2
businessDF = parse_json(business_path)
with open('GA_restaurants_indices.json') as f:
  GA_restaurant_dict = json.load(f)
GA_ID2Rest_dict = {}
for key in GA_restaurant_dict:
  GA_ID2Rest_dict[GA_restaurant_dict[key]] = key

businessDF = businessDF[businessDF['business_id'].isin(GA_restaurant_dict)]
reviewDF = pd.read_csv("GA.csv")
reviewBusinessIDs = reviewDF['business_id'].tolist()
reviewBusinessIDSet = set(reviewBusinessIDs)
reviewBusinessIDs = list(reviewBusinessIDSet)
restaurantCount = len(reviewBusinessIDs)
randomList = random.sample(range(0,restaurantCount), 50)
for i, index in enumerate(randomList):
  randomList[i] = reviewBusinessIDs[index]

reader = Reader(rating_scale=(1, 5))
totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
trainset = totalDataset.build_full_trainset()
# trainset, testset = train_test_split(totalDataset, test_size=.1)
algo = SVD(n_factors = 150, n_epochs = 5, lr_all = 0.01, reg_all = 0.02)
algo.fit(trainset)

# predictions = algo.test(testset)
# accuracy.rmse(predictions)

tsne = TSNE(n_components=2, n_iter=500, verbose=3, random_state=6)
businessEmbedding = tsne.fit_transform(algo.qi)
embeddingDF = pd.DataFrame(columns=['x', 'y'], data=businessEmbedding)
# plot 1
# embeddingDF.plot.scatter('x', 'y')
# plt.show()

# plot 2
modelIndexList = []
subsetDF = pd.DataFrame(columns=['x', 'y', 'title'])
for index in randomList:
  modelIndexList.append(trainset.to_inner_iid(index))
for i, index in enumerate(modelIndexList):
  row = embeddingDF.iloc[index]
  new_row = {'x': row['x'], 'y': row['y'], 'title': businessDF[businessDF['business_id'] == GA_ID2Rest_dict[randomList[i]]]['name']}
  subsetDF = subsetDF.append(new_row, ignore_index=True)
fig = px.scatter(
    subsetDF, x='x', y='y', text='title',
    )
fig.show()
report = dp.Report(dp.Plot(fig) ) #Create a report
report.publish(name='Yelp Embedding 1', open=True, visibility='PUBLIC')