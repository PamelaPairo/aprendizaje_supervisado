#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


URL_TRAIN_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_train.csv"
URL_TEST_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_test.csv"


df_train = pd.read_csv(URL_TRAIN_DATA)
df_test = pd.read_csv(URL_TEST_DATA)


# In[ ]:


X_train_total = df_train.drop(["Customer","TravelInsurance", "GraduateOrNot"], axis=1)
y_train_total = df_train["TravelInsurance"]


# In[ ]:


seed=0
X_train, X_valid, y_train, y_valid = train_test_split(X_train_total,
                                                      y_train_total,
                                                      test_size=0.2,
                                                      random_state=seed)


# In[ ]:


X_test = df_test.drop(["Customer", "GraduateOrNot"], axis=1)


# In[ ]:


numerical_cols = X_train_total.select_dtypes(
    include=['float64', 'int64']).columns

categorical_cols = X_train_total.select_dtypes(include=['object']).columns

vars_to_scale = ["Age", "AnnualIncome", "FamilyMembers"]


# In[ ]:


preprocessor_naive = ColumnTransformer(
    [("encoder", OneHotEncoder(), categorical_cols),
     ("scaler", MinMaxScaler(), vars_to_scale)],# daba un error que decia que no acepta valores negativos en train, asi que lleve los valores a una escala de 0-1
    remainder="passthrough")

pipe_naive = Pipeline([
    ("preprocessor", preprocessor_naive),
    ("naive", MultinomialNB())
])


# In[ ]:


param_naive = {
    'naive__alpha': [0.5, 0.4, 0.3, 0.01, 1],
    'naive__fit_prior': [1, 0],
    'naive__class_prior': [None, [0.5, 0.5], [0.6426, 0.3573]]
}

clf_naive = GridSearchCV(pipe_naive, param_grid=param_naive, scoring="f1", cv=3)
clf_naive.fit(X_train, y_train)


# In[ ]:


clf_naive.best_params_


# In[ ]:


y_train_pred_naiv = clf_naive.best_estimator_.predict(X_train)
y_val_pred_naiv = clf_naive.best_estimator_.predict(X_valid)
y_test_pred_naiv = clf_naive.best_estimator_.predict(X_test)


# In[ ]:


text = "Naive Bayes - Reporte de clasificación del conjunto de validation"
print(len(text)*"=")
print(text)
print(len(text)*"=")
print(classification_report(y_valid, y_val_pred_naiv))

