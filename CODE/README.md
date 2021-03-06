# EECS 476 Final Project
This readme is a brief tutorial on how to run the project and re-produce the results in the report. In case of any question during running, please kindly contact chenqy@umich.edu and yilegu@umich.edu.

# Pre-requisite:
1. Dataset: Yelp dataset (https://www.yelp.com/dataset), unzipped and put in the directory ./dataset. In ./dataset there should be five .json files. For example, the path to one of the .json file will be ./dataset/yelp_academic_dataset_business.json. While we have provided a subset of the dataset in the ./dataset directory and taking 10000 samples from each of .json file, because of the data filtering scheme of our code requires quite many data to produce the filtered result, the filtered result got in GA.csv or *_indices.json will have nothing in it, so we recommend downloading the generated csv and json files by visiting our shared folder on Google Drive (https://drive.google.com/drive/folders/1xzkQkUKBEKK87HafFh0gs1etXGCUaE_7?usp=sharing).

2. Packages: Many of these are quite common packages and are likely to have been installed on users' platforms. In the case of missing a package, pip should install them fine.
	json
	numpy
	pandas
	collections
	matplotlib
	dataprocessing
	surprise
	lightfm
	sklearn
	plotly
	datapane

# How to run the project:
## MAKEFILE Usage
Please do not run make commands on MAKEFILE since it is written in a bash style. Instead, please run "./MAKEFILE.sh" to run. You may need to run "chmod +x MAKEFILE.sh" to be able to run it as a script. Such way of running the project is equivalent to following the steps from 1 to 9 (i.e., if you run "./MAKEFILE.sh", then you won't need to go through step 1 to 9 one by one). Also note that some of the python files (specified in the below section) only define useful functions for users to run. In order to run the project correctly, the user has to define the functions he or she wants to run in the corresponding python file. By downloading the generated .csv and .json file in the Google Drive specified above, you can actually start from step 4 and skip steps 1-3.

## Step-to-Step Usage
1. Run "python3 filter_GA_restaurants_and_users.py" to get two files, "GA_restaurants.json" and "GA_users.json". This step filters our data of interest and limit the user and business data to a reasonable range as specified in the report. Note that the processed json files are included as part of the dataset.

2. Run "python3 GA_resturants_and_users_to_index.py" to create a mapping of users and businesses to their corresponding indices. You should have "GA_restaurants_indices.json" and "GA_users_indices.json" by then. Note that the processed json files are included as part of the dataset.

3. "python3 dataprocessing.py" defines several useful functions for processing the filtered data. Run function "GA_processing()" to create the rating matrix R. It should generate a file called "GA.csv". Note that due to the size of generated csv file, it is not included in the zip file.

4. Run "python3 baseline_model.py" to use the baseline model and get corresponding metrics like RMSE. The results will be written to a file called "baseline_model_result.txt". 

5. Run "python3 SVD_model.py" to search for hyperparameters and get the best set of hyperparameters for SVD model. Training results with RMSE metrics will be written to a file called "SVD_model_result.txt".

6. Run "python3 NMF_model.py" to search for hyperparameters and get the best set of hyperparameters for NMF model. Training results with RMSE metrics will be written to a file called "NMF_model_result.txt".

7. Run "python3 lightFM_model_FeatureBased.py" to train the feature-based lightFM model and get corrsponding metrics like AUC score. Additionally, the code will generate the most similar categories for "American (Traditional)", "Japanese" and "Cafes".

8. When needed, run "python3 prediction_recall_k.py" with the best set of hyperparamters to get metrics like AUC score for baseline, SVD and NMF models.

9. "python3 visualization.py" defines several functions to generate the plots in the report. Run function "generate_embeddings()" to generate the 2D embedding for SVD model.

