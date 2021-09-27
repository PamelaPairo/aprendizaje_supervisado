#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[ ]:


import pandas as pd
from utils import (make_pipeline, X_train, X_val, save_predictions, y_train,
                   y_val, X_train_total, y_train_total)
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


# In[ ]:


URL_TRAIN_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_train.csv"
URL_TEST_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_test.csv"


df_train = pd.read_csv(URL_TRAIN_DATA)
df_test = pd.read_csv(URL_TEST_DATA)


# In[ ]:


import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:


seed = 0

X_train_total = df_train.drop(["Customer","TravelInsurance", "GraduateOrNot"], axis=1)
y_train_total = df_train["TravelInsurance"]


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train_total,
                                                      y_train_total,
                                                      test_size=0.2,
                                                      random_state=seed)
X_test = df_test.drop(["Customer", "GraduateOrNot"], axis=1)


# In[ ]:


numerical_cols = X_train_total.select_dtypes(
    include=['float64', 'int64']).columns

categorical_cols = X_train_total.select_dtypes(include=['object']).columns

vars_to_scale = ["Age", "AnnualIncome", "FamilyMembers"]

preprocessor = ColumnTransformer(
    [("encoder", OneHotEncoder(), categorical_cols),
     ("scaler", StandardScaler(), vars_to_scale)],
    remainder="passthrough")


# In[ ]:


pipe_rfc = Pipeline([
    ("preprocessor", preprocessor),
    ("rfc", RandomForestClassifier(random_state=seed))
])


# In[ ]:


params_rfc = {
    "rfc__max_depth": Integer(5, 40),
    "rfc__criterion":Categorical(['gini', 'entropy']),
    "rfc__min_samples_split": Integer(3,20),
    "rfc__min_samples_leaf":Integer(1,25),
    "rfc__bootstrap": Categorical([False]),
    "rfc__n_estimators": Integer(20, 150)
}

opt = BayesSearchCV(pipe_rfc, params_rfc, scoring="f1", cv=3)
#opt.fit(X_train_total, y_train_total)


# In[ ]:


opt.fit(X_train, y_train)


# In[ ]:


print(opt.score(X_valid, y_valid))


# In[ ]:


opt.best_estimator_


# In[ ]:


y_val_pred = opt.best_estimator_.predict(X_valid)
print(classification_report(y_valid, y_val_pred))


# In[ ]:


from os import makedirs
DIR_SUBMISSIONS = "submissions"

def save_predictions(model, filename):
    test_id = df_test["Customer"]
    test_pred = model.predict(df_test.drop(columns=["Customer", "GraduateOrNot"]))

    submission = pd.DataFrame(list(zip(test_id, test_pred)),
                              columns=["Customer", "TravelInsurance"])
    
    makedirs(DIR_SUBMISSIONS, exist_ok=True)
    submission.to_csv(f"{DIR_SUBMISSIONS}/{filename}",
                      header=True,
                      index=False)


# In[ ]:


save_predictions(opt.best_estimator_, "random_forest.csv")


# ## Con todo X_train_total

# In[ ]:


opt_total = BayesSearchCV(pipe_rfc, params_rfc, scoring="f1", cv=3)
opt_total.fit(X_train_total, y_train_total)


# In[ ]:


opt_total.best_estimator_


# In[ ]:


save_predictions(opt.best_estimator_, "random_forest_total.csv")


# In[ ]:




