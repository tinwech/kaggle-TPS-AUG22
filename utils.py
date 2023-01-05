from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from config import *

impute_features = ['measurement_3', 'measurement_4',
                    'measurement_5','measurement_6','measurement_7',
                    'measurement_8','measurement_9','measurement_17']

iter_imputer = IterativeImputer(max_iter=20)
simple_imputer = SimpleImputer(strategy='median')

def impute(data, fit=False):
    if fit:
        iter_imputer.fit(data[impute_features])
        simple_imputer.fit(data)

    data[impute_features] = iter_imputer.transform(data[impute_features])
    data[:] = simple_imputer.transform(data)
    return data

def add_features(df):
    df['missing_3'] = df['measurement_3'].isna()
    df['missing_5'] = df['measurement_5'].isna()
    df['missing_loading'] = df['loading'].isna()
    df['area'] = df['attribute_2'] * df['attribute_3']
    return df

def pre_processing(*df):
    df = list(df)
    for i in range(len(df)):
        # drop features
        df[i].drop(columns=drop_features, inplace=True)

        # add features
        df[i] = add_features(df[i])

        # imputation
        if i == 0:
            df[i] = impute(df[i], fit=True)
        else:
            df[i] = impute(df[i])

        # select features
        df[i] = df[i][select_features]

    if len(df) == 1:
        return df[0]
    return df
