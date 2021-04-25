import json

with open('GA_restaurants.json') as f:
  GA_restaurant_list = json.load(f)
  count = 0
  restaurant_dict = {}
  for restaurant in GA_restaurant_list:
    restaurant_dict[restaurant] = count
    count += 1
  out_file = open("GA_restaurants_indices.json", "w") 
  json.dump(restaurant_dict, out_file, indent = 6)
  out_file.close()

with open('GA_users.json') as f:
  GA_users_list = json.load(f)
  count = 0
  user_dict = {}
  for user in GA_users_list:
    user_dict[user] = count
    count += 1
  out_file = open("GA_users_indices.json", "w") 
  json.dump(user_dict, out_file, indent = 6)
  out_file.close()

