# %%
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# %%

data = pd.read_csv(
    'c:\\Users\\jefma\\Documents\\GitHub\\DataTrain\\data\\teste.csv')

# %%
data.info()

# %%
# retirando nulos de estadocivilmae
data.estadocivilmae.value_counts()
data.estadocivilmae[data.estadocivilmae.isnull()] = 'desconhecido'

# %%
# retirando nulos de catprenatal
data.catprenatal.value_counts()
data.catprenatal[data.catprenatal.isnull()] = 'desconhecido'


# %%
# retirando nulos de qtdsemanas
data.qtdsemanas.value_counts()
data.qtdsemanas[data.qtdsemanas.isnull()] = data.qtdsemanas.mean()


# %%
# retirando nulos de tipoparto
data.tipoparto.value_counts()
data.tipoparto[data.tipoparto.isnull()] = 'desconhecido'

# %%
# retirando nulos de peso
data.peso.value_counts()
data.peso[data.peso.isnull()] = data.peso.mean()
# %%
# retirando nulos de malformacao
data.malformacao.value_counts()
data.malformacao[data.malformacao.isnull()] = 'desconhecido'

# %%
# retirando nulos de sexo
data.sexo.value_counts()
data.sexo[data.sexo.isnull()] = 'desconhecido'

# %%
# retirando nulos de apgar1
data.apgar1.value_counts()
data.apgar1[data.apgar1.isnull()] = data.apgar1.mean()

# %%
# retirando nulos de apgar5
data.apgar5.value_counts()
data.apgar5[data.apgar5.isnull()] = data.apgar5.mean()
# %%

# %%
data.to_csv(
    'c:\\Users\\jefma\\Documents\\GitHub\\DataTrain\\data\\nnTest.csv', index=False)

# %%')
# %%

# %%
