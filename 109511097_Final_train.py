import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from utils import pre_processing, load_data
from config import *


def train():
    X, y, _ = load_data()
    avg_score = 0
    group_kfold = GroupKFold(n_splits=n_splits)
    for i, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=X['product_code'])):
        # training set
        X_train = X.loc[train_idx, :]
        y_train = y.loc[train_idx]

        # validation set
        X_val = X.loc[val_idx, :]
        y_val = y.loc[val_idx]
        
        # pre-processing
        X_train, X_val = pre_processing(X_train, X_val)

        # convert dataframe to numpy
        X_train = X_train.values
        y_train = y_train.values
        X_val = X_val.values
        y_val = y_val.values

        # fit model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        # evaluation
        score = clf.score(X_val, y_val)
        avg_score += score / n_splits
        print(f'validation {i} score: {score}')

    print('average score:', avg_score)

def save_model():
    X, y, _ = load_data()

    X = pre_processing(X)
    
    X = X.values
    y = y.values

    clf = LogisticRegression()
    clf.fit(X, y)

    # save model
    filename = f'model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print(f'Saved model to {filename}')

if __name__ == '__main__':
    train()
    save_model()