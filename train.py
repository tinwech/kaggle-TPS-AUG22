import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from utils import pre_processing
from config import *

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

group_kfold = GroupKFold(n_splits=n_splits)

avg_score = 0
X = train_df.drop(columns=['failure'])
y = train_df['failure']
for i, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=X['product_code'])):
    # training set
    X_train = X.loc[train_idx, :]
    y_train = y.loc[train_idx]

    # pre-processing
    X_train = pre_processing(X_train)

    # convert dataframe to numpy
    X_train = X_train.values
    y_train = y_train.values

    # fit model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # save model
    filename = f'model_{i}.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print(f'Saved model to {filename}')
