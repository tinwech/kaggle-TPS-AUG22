import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from utils import pre_processing
from config import *

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

group_kfold = GroupKFold(n_splits=n_splits)

y_pred = []
X = train_df.drop(columns=['failure'])
y = train_df['failure']
for i, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=X['product_code'])):
    # training set (in order to have the same imputation)
    X_train = X.loc[train_idx, :]

    # testing set
    X_test = test_df.copy()

    # pre-processing
    X_train, X_test = pre_processing(X_train, X_test)

    # convert dataframe to numpy
    X_test = X_test.values

    # load model
    clf = pickle.load(open(f'model_{i}.pkl', 'rb'))

    # predict testing set
    y_pred.append(clf.predict_proba(X_test)[:, 1])


# save prediction on testing set
y_pred = np.array(y_pred)
y_pred = np.mean(y_pred, axis=0)
pred_df = pd.DataFrame(y_pred, columns=['failure'])
submission_df = pd.concat([test_df, pred_df], axis=1)[['id', 'failure']]
submission_df.to_csv('submission.csv', index=False)
print('Saved submission to submission.csv')