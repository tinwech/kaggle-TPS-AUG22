import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from utils import pre_processing
from config import *

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

group_kfold = GroupKFold(n_splits=n_splits)

avg_score = 0
X = train_df.drop(columns=['failure'])
y = train_df['failure']
for i, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=X['product_code'])):
    # training set (in order to have the same imputation)
    X_train = X.loc[train_idx, :]

    # validation set
    X_val = X.loc[val_idx, :]
    y_val = y.loc[val_idx]

    # testing set
    X_test = test_df.copy()

    # pre-processing
    X_train, X_val, X_test = pre_processing(X_train, X_val, X_test)

    # convert dataframe to numpy
    X_val = X_val.values
    y_val = y_val.values
    X_test = X_test.values

    # load model
    clf = pickle.load(open(f'model_{i}.pkl', 'rb'))

    # evaluate model on validation set
    score = clf.score(X_val, y_val)
    avg_score += score / n_splits
    print(f'model_{i} validation score: {score}')

# average score
print(f'average score: {avg_score}')
