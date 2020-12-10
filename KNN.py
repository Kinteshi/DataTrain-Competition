# %%
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


# %%
data = pd.read_csv(
    'c:\\Users\\jefma\\Documents\\GitHub\\DataTrain\\data\\treino.csv'
)

null_replacers = [
    data.idademae.mean(),
    'desconhecido',  # estadocivil
    'desconhecido',  # catprenatal
    data.qtdsemanas.mean(),
    'desconhecido',  # tipoparto
    data.peso.mean(),
    'desconhecido',  # malformacao
    'desconhecido',  # sexo
    data.apgar1.mean(),
    data.apgar5.mean()
]


# null_replacers = [
#     data.idademae.mean(),
#     data.estadocivilmae.mode()[0],  # estadocivil
#     data.catprenatal.mode()[0],  # catprenatal
#     data.qtdsemanas.mean(),
#     data.tipoparto.mode()[0],  # tipoparto
#     data.peso.mean(),
#     data.malformacao.mode()[0],  # malformacao
#     data.sexo.mode()[0],  # sexo
#     data.apgar1.mean(),
#     data.apgar5.mean()
# ]

y = data.obito
data.drop(columns=['id', 'obito'], inplace=True)

for i, col in enumerate(data.columns):
    data[col][data[col].isnull()] = null_replacers[i]

ohe = OneHotEncoder(drop='first')
ohe.fit(data.values[:, [1, 2, 4, 6, 7]])
ohe_data = ohe.transform(
    data.values[:, [1, 2, 4, 6, 7]])

ohe_data = pd.DataFrame.sparse.from_spmatrix(
    ohe_data, columns=ohe.get_feature_names())

X = data.drop(columns=['estadocivilmae', 'tipoparto',
                       'malformacao', 'sexo', 'catprenatal'])

X = pd.concat([X, ohe_data], axis='columns')

X.info()
feature_name = X.columns

mms = MinMaxScaler()
mms.fit(X)
X = mms.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


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

############################################


# model.evaluate(X_test, y_test)

###########################################################
# %%


# %%
acu = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    c = dict({'k': i})
    # Predicting results using Test data set
    pred = knn.predict(X_test)
    c['test'] = accuracy_score(pred, y_test)

    # train
    pred = knn.predict(X_train)
    c['train'] = accuracy_score(pred, y_train)
    acu.append(c)
acu
# %%
sorted(acu, key=lambda x: x['train'], reverse=True)[:3]
# %%
knn = KNeighborsClassifier(
    n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
accuracy_score(pred, y_test)

pred = knn.predict(X_test)
auc = roc_auc_score(y_test, pred)
auc

# %%
# train
