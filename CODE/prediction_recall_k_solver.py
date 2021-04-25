"""A solver for prediction@k and recall@k of Surprise models, modified from official documentation:
https://surprise.readthedocs.io/en/stable/FAQ.html?highlight=top#how-to-compute-precision-k-and-recall-k
"""
from collections import defaultdict
import pandas as pd
from surprise import Dataset
from surprise import SVD
from surprise import BaselineOnly
from surprise.model_selection import KFold
from surprise import Reader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def split_prediction(predictions, threshold=3.5):
    pred_df = pd.DataFrame(predictions)
    pred_df['r_ui'].where(pred_df['r_ui']>threshold, 1, inplace=True)
    pred_df['r_ui'].where(pred_df['r_ui']<=threshold, 0, inplace=True)
    pred_df['est'].where(pred_df['est']>threshold, 1, inplace=True)
    pred_df['est'].where(pred_df['est']<=threshold, 0, inplace=True)

    return pred_df['r_ui'], pred_df['est'], 

def plot_roc (true_r, est):
    fpr, tpr, thresholds = roc_curve(true_r, est)
    auc_ = auc(fpr, tpr)
    plt.plot(fpr,tpr, '-')
    plt.ylabel("FPR/Precision Scores")
    plt.xlabel("TPR/Recall Scores")
    plt.title("ROC Scores, with AUC = " + str(round(auc_, 2)))
    plt.show()


reviewDF = pd.read_csv("GA.csv")
reader = Reader(rating_scale=(1, 5))
totalDataset = Dataset.load_from_df(reviewDF[['user_id', 'business_id', 'stars']], reader)
kf = KFold(n_splits=5)
algo = SVD(n_factors = 100, n_epochs = 15, lr_all = 0.02, reg_all = 0.2)
# bsl_options = {'method': 'sgd',
#               'learning_rate': .00005,
#               }
# algo = BaselineOnly(bsl_options=bsl_options)

precision_list = []
recall_list = []

for i, (trainset, testset) in enumerate(kf.split(totalDataset)):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3)

    # Precision and recall can then be averaged over all users
    precision_list.append(sum(prec for prec in precisions.values()) / len(precisions))
    recall_list.append(sum(rec for rec in recalls.values()) / len(recalls))
    print("=== Fold ", i, "precision: ", precision_list[i])
    print("=== Fold ", i, "recall: ", recall_list[i])

print("Final mean precision: ", sum(precision_list)/len(precision_list))
print("Final mean recall: ", sum(recall_list)/len(recall_list))
true_r, est = split_prediction(predictions)
plot_roc(true_r, est)