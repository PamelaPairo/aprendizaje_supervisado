# -*- coding: utf-8 -*-
# # Diplomatura en Ciencias de Datos, Aprendizaje Automático y sus Aplicaciones
#
# Autores: Matías Oria, Antonela Sambuceti, Pamela Pairo, Benjamín Ocampo
# ## Mejores 3 Modelos de la Competencia
# Se trabajó sobre los siguientes modelos:
#
# - `xgb`: XGBoost.
# - `lgr`: Regresión logistica.
# - `svm`: Support Vector Machines.
# - `random_forest`: Random Forest.
# - `dtree`: Decision Tree.
# - `sgd`: SGD Classifier para regresión logistica.
# - `nnet`: Neural Networks (Sklearn).
# - `nnet_keras`: Neural Networks (Keras).
# - `naive_bayes`: Naive Bayes.
#
# De los cuales los `xgb`, `svm`, y `rndforest` lograron los mejores resultados
# para el conjunto de *test* público en la competencia de Kaggle.
#
# La implementación y busqueda de hiperparametros del resto de modelos puede
# encontrarse en una notebook dedicada para cada uno.
# ### Pipeline
# El pipeline planteado durante el proceso de aprendizaje consiste de 3 capas.
#
# - Discretización para las variables continuas `Age`, y `AnnualIncome`.
# - Códificación one-hot para el resto de variables categoricas.
# - Modelo.
#
# Donde las primeras dos son de preprocesamiento, mientras que la tercera es el
# predictor utilizado.
from utils import (make_pipeline, save_predictions, X_train_total,
                   y_train_total, X_train, X_val, y_train, y_val, X_test,
                   df_test, seed, DIR_SUBMISSIONS)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from os import makedirs
import pandas as pd

# ### 1º XGBoost
# Para este modelo se optó por realizar una busqueda de hiperparámetros
# utilizando todas las filas y columnas del conjunto de entrenamiento por medio
# de una Busqueda Bayesiana (`Bayesian Search`) donde se especifica un rango o
# espacio sobre los parametros que se desean encontrar. Este espacio se recorre
# de manera apropiada recordando iteraciones pasadas de busqueda hasta encontrar
# el modelo óptimo. Algo a recalcar es que no solamente se ajustó la signature
# del clasificador, si no que también se intentó encontrar distintas
# discretizaciones posibles durante el preprocesamiento. 
#
# Resultado en Kaggle f1-score: 0.80898
# +
from xgboost import XGBClassifier

xgboost = make_pipeline(XGBClassifier())
params_bayes = {
    "preprocessor__discretizer__n_bins": Integer(2, 25),
    "model__objective": Categorical(["binary:logistic"]),
    "model__n_estimators": Integer(10, 500),
    "model__gamma": Real(1e-6, 1),
    "model__max_depth": Integer(4, 20),
    "model__learning_rate": Real(1e-4, 1),
    "model__alpha": Real(1e-4, 1),
    "model__booster": Categorical(["gbtree", "gblinear", "dart"]),
    "model__colsample_bytree": Real(.5, 1),
    "model__subsample": Real(.6, 1),
    "model__eval_metric": Categorical(["logloss"]),
    "model__use_label_encoder": Categorical([False]),
}
opt = BayesSearchCV(xgboost, params_bayes, scoring="f1")
opt.fit(X_train_total, y_train_total)
save_predictions(opt.best_estimator_, "xgb.csv")
opt.best_params_
# -
# ### 2º SVM
# El segundo modelo consta de una busqueda de hiperparametros por medio de una
# Grid Search recorriendo sobre 4 kernels posibles, `linear`, `sigmoid`, `poly`,
# `rbf`. A diferencia de `xgb`, en este caso se optó por dividir los datos en
# train y validación para obtener métricas del modelo sobre este último conjunto.
# Una vez realizada la busqueda de hiperparametros sobre el conjunto de datos de
# entrenamiento, se utilizan los mejores hiperparametros para realizar un último
# ajuste con `train` + `validation`.
#
# Resultado en Kaggle f1-score: 0.80459
# +
from sklearn.svm import SVC
svm = make_pipeline(SVC())

params = {
    "model__kernel": ["linear", "sigmoid", "poly", "rbf"],
    "model__gamma": ["scale", "auto"],
    "model__degree": [2, 3, 4],
    "model__coef0": [.001, .01, 0, 1],
    "model__tol": [1e-2, 1e-3, 1e-4],
    "model__C": [1, 0.1, 0.01, 0.001, 0.0001, 10],
    "model__class_weight": ["balanced"]
}
clf = GridSearchCV(svm, param_grid=params, scoring="f1")
clf.fit(X_train, y_train)
y_val_pred = clf.best_estimator_.predict(X_val)
print(classification_report(y_val, y_val_pred))
# -
# Refit with validation data
best_model = make_pipeline(
    SVC(
        **{
            key.removeprefix("model__"): value
            for key, value in clf.best_params_.items()
        }))
best_model.fit(X_train_total, y_train_total)
save_predictions(best_model, "svm.csv")
# ### 3º Random Forest
# En este último caso, se optó por utilizar Busqueda Bayesiana dividiendo los datos de entrenamiento y se obtuvo el score F1 mas alto eliminando la variable `GraduateOrNot`-
#
# Resultado en Kaggle f1-score: 0.80459
# +
from sklearn.ensemble import RandomForestClassifier

X_train_total = X_train_total.drop(["GraduateOrNot"], axis=1)
X_train = X_train.drop(["GraduateOrNot"], axis=1)
X_test = X_test.drop(["GraduateOrNot"], axis=1)
X_val = X_val.drop(["GraduateOrNot"], axis=1)

# +
vars_to_ohe = X_train_total.select_dtypes(include=['object']).columns

vars_to_scale = ["Age", "AnnualIncome", "FamilyMembers"]

preprocessor = ColumnTransformer(
    [("encoder", OneHotEncoder(), vars_to_ohe),
     ("scaler", StandardScaler(), vars_to_scale)],
    remainder="passthrough")
# -

pipe_rfc = Pipeline([
    ("preprocessor", preprocessor),
    ("rfc", RandomForestClassifier(random_state=seed))
])

# +
params_rfc = {
    "rfc__max_depth": Integer(5, 40),
    "rfc__criterion":Categorical(['gini', 'entropy']),
    "rfc__min_samples_split": Integer(3,20),
    "rfc__min_samples_leaf":Integer(1,25),
    "rfc__bootstrap": Categorical([False]),
    "rfc__n_estimators": Integer(20, 150)
}

opt = BayesSearchCV(pipe_rfc, params_rfc, scoring="f1", cv=3)
opt.fit(X_train, y_train)
# -

opt.best_estimator_

y_val_pred = opt.best_estimator_.predict(X_val)
print(classification_report(y_val, y_val_pred))


def save_predictions(model, filename):
    test_id = df_test["Customer"]
    test_pred = model.predict(df_test.drop(columns=["Customer", "GraduateOrNot"]))

    submission = pd.DataFrame(list(zip(test_id, test_pred)),
                              columns=["Customer", "TravelInsurance"])
    
    makedirs(DIR_SUBMISSIONS, exist_ok=True)
    submission.to_csv(f"{DIR_SUBMISSIONS}/{filename}",
                      header=True,
                      index=False)


save_predictions(opt.best_estimator_, "random_forest.csv")


