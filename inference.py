import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from utils import pre_processing, load_data
from config import *

def inference():
    X_train, y_train, test_df = load_data()

    # testing set
    X_test = test_df.copy()

    # pre-processing
    X_train, X_test = pre_processing(X_train, X_test)

    # convert dataframe to numpy
    X_test = X_test.values

    # load model
    clf = pickle.load(open(f'model.pkl', 'rb'))

    # predict testing set
    y_pred = clf.predict_proba(X_test)[:, 1]

    # save prediction
    pred_df = pd.DataFrame(y_pred, columns=['failure'])
    submission_df = pd.concat([test_df, pred_df], axis=1)[['id', 'failure']]
    submission_df.to_csv('submission.csv', index=False)
    print('Saved submission to submission.csv')

if __name__ == '__main__':
    inference()
