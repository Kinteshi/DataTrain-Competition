# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# %%
sns.set()
sns.set_theme(style='ticks', color_codes=True)


# %%
# Load Train Data
data = pd.read_csv('data/treino.csv')


# %%
data.info()


# %%
data.describe()


# %%
# sns.histplot(data=data, x='idademae', hue='obito', kde=True)


# %%
data.estadocivilmae[data.estadocivilmae.isnull()] = 'nao informado'
# sns.histplot(data=data, x='estadocivilmae', hue='obito')


# %%
data.catprenatal[data.catprenatal.isnull()] = 'nao informado'
# sns.histplot(data=data, x='catprenatal', hue='obito')


# %%
# sns.histplot(data=data, x='qtdsemanas', hue='obito', kde=True, binwidth=2)


# %%
data.tipoparto[data.tipoparto.isnull()] = 'desconhecido'
# sns.histplot(data=data, x='tipoparto', hue='obito')


# %%
# sns.histplot(data=data, x='peso', hue='obito', kde=True)


# %%
data.malformacao[data.malformacao.isnull()] = 'desconhecido'
# sns.histplot(data=data, x='malformacao', hue='obito')


# %%
data.sexo[data.sexo.isnull()] = 'desconhecido'
# sns.histplot(data=data, x='sexo', hue='obito')


# %%
# sns.histplot(data=data, x='apgar1', hue='obito', kde=True)


# %%
# sns.histplot(data=data, x='apgar5', hue='obito', kde=True)


# %%
# data.corr()


# %%

# for ycol in data.columns[1:]:
#     for xcol in data.columns[1:]:
#         print(f'{ycol}_{xcol}')
#         if xcol == ycol:
#             continue
#         sns.catplot(data=data, x=xcol, y=ycol, hue='obito').savefig(
#             f'allbyall_plots/{ycol}_{xcol}.png')

# %%
