# %%
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as numpy

# %%
data = pd.read_csv(
    'data/treino.csv'
)
data.drop(columns=['id', 'obito'], inplace=True)

data_not_none = data.dropna()
data_none = data.iloc[data.isnull().any(axis=1).values]

# %%

ohe = OneHotEncoder(drop='first')
ohe.fit(data_not_none.values[:, [1, 2, 4, 6, 7]])
ohe_data = ohe.transform(
    data.values[:, [1, 2, 4, 6, 7]])

ohe_data = pd.DataFrame.sparse.from_spmatrix(
    ohe_data, columns=ohe.get_feature_names())

X = data_not_none.drop(columns=['estadocivilmae', 'tipoparto',
                       'malformacao', 'sexo', 'catprenatal'])

X = pd.concat([X, ohe_data], axis='columns')


def testset_preprocessing(data):
    for i, col in enumerate(data.columns):
        data[col][data[col].isnull()] = null_replacers[i]
    ohe_data = ohe.transform(
        data.values[:, [1, 2, 4, 6, 7]])
    ohe_data = pd.DataFrame.sparse.from_spmatrix(
        ohe_data, columns=ohe.get_feature_names())
    X = data.drop(columns=['estadocivilmae', 'tipoparto',
                           'malformacao', 'sexo', 'catprenatal'])
    X = pd.concat([X, ohe_data], axis='columns')
    X = mms.transform(X)
    return X

# %%


knn = NearestNeighbors(n_neighbors=3, algorithm='brute)
knn.fit(data_not_none)


# %%
# Predicting results using Test data set
# pred = knn.predict(X_test)
# accuracy_score(pred, y_test)

# # %%
# # train
# pred = knn.predict(X_train)
# accuracy_score(pred, y_train)
