# %%
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model
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


# %%
i = Input(shape=X.shape[1])
x = Dense(64, activation='sigmoid')(i)
x = Dense(128, activation='sigmoid')(x)
x = Dense(64, activation='sigmoid')(x)
x = Dense(32, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'],)

# %%
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# %%
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

# %%
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

# %%
plt.plot(r.history['auc'], label='auc')
plt.plot(r.history['val_auc'], label='val_auc')
plt.legend()
# %%
model.evaluate(X_test, y_test)

###########################################################
# %%
test = pd.read_csv(
    'c:\\Users\\jefma\\Documents\\GitHub\\DataTrain\\data\\teste.csv')

ids = test.id
test.drop(columns=['id'], inplace=True)
tX = testset_preprocessing(test)
# %%
yhat = model.predict(tX).flatten()

# %%
zipped = np.zeros(shape=(len(yhat), 2), dtype='O')

for i in range(len(yhat)):
    zipped[i, 0] = str(ids[i])
    zipped[i, 1] = float(yhat[i])

out = pd.DataFrame(zipped, columns=['id', 'obito'])

out.to_csv('data/out.csv', index=False)

# %%
