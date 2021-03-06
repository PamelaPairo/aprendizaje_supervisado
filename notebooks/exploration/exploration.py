# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: 'Python 3.9.7 64-bit (''diplodatos-supervised'': conda)'
#     name: python397jvsc74a57bd01fbf230ed8a49891bdc325a868414da3a93ccc603acb0edf0a274c21ede9f486
# ---

# %% [markdown]
# Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo

# %% [markdown]
# ## Importación de librerias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %% [markdown]
# ## Lectura del dataset

# %%
URL_TRAIN_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_train.csv"
URL_TEST_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_test.csv"

df_train = pd.read_csv(URL_TRAIN_DATA)
df_test = pd.read_csv(URL_TEST_DATA)
# %%
df_train.info()
# %%
df_test.info()
# %%
df_train
# %%
df_test
# %% [markdown]
# ## Análisis descriptivo
# %%
df_train.describe().round()
# %% [markdown]
# ### Variable objetivo o target: Travel Insurance
# %%
df_train["TravelInsurance"].value_counts()
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=df_train, x="TravelInsurance")
# %% [markdown]
# ### Correlación variables

# %%
corr = df_train[["TravelInsurance", "AnnualIncome", "Age"]].corr().round(2)
corr = corr[['TravelInsurance']]
corr.loc[:, 'abs_corr'] = np.abs(corr['TravelInsurance'])
corr.sort_values(by='abs_corr', ascending=False)
# %%
plt.figure(figsize=(10, 10))
corr = df_train.corr()
sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True,
            cmap='coolwarm')
plt.show()
# %% [markdown]
# ### Variable explicativa: Age

# %% [markdown]
# En primer lugar, observamos la distribución de la variable Age en el df_train.
# %%
sns.countplot(data=df_train, x="Age", hue="TravelInsurance")
# %%
df_train[["TravelInsurance", "Age"]].groupby("TravelInsurance").describe()
# %% [markdown]
# Podemos observar que se encuentran fluctuaciones con respecto a la edad.
# %% [markdown]
# ### Variable explicativa: AnnualIncome

# %% [markdown]
# En segundo lugar, observamos la distribución de la variable Annual Income en el df_train.

# %%
fig = plt.figure(figsize=(8, 8))
sns.boxenplot(data=df_train, x="AnnualIncome")
plt.ticklabel_format(style='plain', axis='x')

# %%
fig = plt.figure(figsize=(8, 8))
sns.boxenplot(data=df_train, x="TravelInsurance", y="AnnualIncome")
plt.ticklabel_format(style='plain', axis='y')

# %%
(df_train[["TravelInsurance",
           "AnnualIncome"]].groupby(["TravelInsurance"]).describe())
# %% [markdown]
# Podemos observar que la distribución de la variable Annual Income se ve afectada en gran medida, al condicionarla por las distintas clases de la variable TARGET. Si bien los valores mínimos y máximos son similares, la media y la mediana difieren considerablemente, como así también el rango intercuantil.
# Es una varible que se considera importante para explicar el comportamiento de Y.

# %% [markdown]
# ### Variable explicativa: Employment Type
# %%
df_train["Employment Type"].value_counts()
pd.crosstab(df_train["TravelInsurance"], df_train["GraduateOrNot"])
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=df_train,
              x="TravelInsurance",
              hue=df_train["Employment Type"])
pd.crosstab(df_train["TravelInsurance"], df_train["Employment Type"])

# %% [markdown]
# ### Variable explicativa: Graduate Or Not

# %%
df_train["GraduateOrNot"].value_counts()
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=df_train,
              x="TravelInsurance",
              hue=df_train["GraduateOrNot"])
# %%
df_train
# %% [markdown]
# ### Variable explicativa: Frequent Flyer

# %%
df_train["FrequentFlyer"].value_counts()

# %% [markdown]
# ### Variable explicativa: Ever Travelled Abroad

# %%
df_train["EverTravelledAbroad"].value_counts()

# %% [markdown]
# ### Variable explicativa: Employment Type

# %%
sns.countplot(data=df_train, x="Employment Type")

# %% [markdown]
# ### Variable explicativa: Family Members
# %%
df_train.columns
# %%
plt.figure(figsize=(10, 8))
sns.histplot(data=df_train,
             x="FamilyMembers",
             hue='TravelInsurance',
             multiple="stack")
# %% [markdown]
# ### Variable explicativa: Chronic Diseases
# %%
plt.figure(figsize=(10, 8))
ax = sns.countplot(data=df_train, x="ChronicDiseases", hue="TravelInsurance")
legend_labels, _ = ax.get_legend_handles_labels()
ax.legend(
    legend_labels,
    ['Not buyed', 'Buyed'],  #ver de cambiar!
    title_fontsize=18,
    fontsize=15,
    bbox_to_anchor=(1, 1),
    title='Travel Insurance')
# %% [markdown]
# ### Variable explicativa: FrequentFlyer
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=df_train, x="TravelInsurance", hue=df_train.FrequentFlyer)

pd.crosstab(df_train["TravelInsurance"], df_train["FrequentFlyer"])
# %% [markdown]
# Se puede observar en este gráfico que si no sos viajero frecuente la cantidad
# de clientes que contratan un seguro es baja, caso contrario la cantidad de
# contratar un seguro es pareja.
# %% [markdown]
# ### Variable explicativa: EverTravelledAbroad
# %%
fig = plt.figure(figsize=(5, 5))
sns.countplot(data=df_train,
              x="TravelInsurance",
              hue=df_train.EverTravelledAbroad)

pd.crosstab(df_train["TravelInsurance"], df_train["EverTravelledAbroad"])