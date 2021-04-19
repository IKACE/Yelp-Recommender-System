import json
import numpy as np

GA_restaurant_dict = {}
GA_user_dict = {}

with open('GA_restaurants_indices.json') as f:
  GA_restaurant_dict = json.load(f)

with open('GA_users_indices.json') as f:
  GA_user_dict = json.load(f)

input_matrix = np.zeros((len(GA_user_dict), len(GA_restaurant_dict)), dtype=np.float)
review_path = "./dataset/yelp_academic_dataset_review.json"
with open(review_path, 'r', encoding='UTF-8') as f:
  lines = f.readlines()
  for line in lines:
    review_json = json.loads(line.strip())
    user_id = review_json["user_id"]
    restaurant_id = review_json["business_id"]
    if user_id in GA_user_dict and restaurant_id in GA_restaurant_dict:
      input_matrix[GA_user_dict[user_id], GA_restaurant_dict[restaurant_id]] = float(review_json["stars"])

np.savetxt("user_restaurant_matrix.txt", input_matrix, delimiter=',')
