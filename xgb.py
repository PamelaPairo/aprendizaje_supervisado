# %%
from utils import (make_pipeline, X_train, X_val, save_predictions, y_train,
                   y_val, X_train_total, y_train_total)
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# %%
pipe = make_pipeline(XGBClassifier())
# %%
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)
# %%
print(classification_report(y_val, y_val_pred))
# %% [markdown]
# ## Fine Tunning
# %%
params = {
    "model__objective": ["binary:logistic"],
    "model__n_estimators": [10, 50, 100, 500],
    "model__gamma": [1e-1, 1e-2, 1e-3],
    "model__max_depth": [4, 6, 8, 10, 20],
    "model__learning_rate": [1e-1, 1e-2, 1e-3],
    "model__alpha": [1e-1, 1e-2, 1e-3],
    "model__booster": ["gbtree", "gblinear", "dart"],
    "model__colsample_bytree": [.5, .6, .7, .8, .9, 1],
    "model__subsample": [.6, .7, .8, .9, 1],
    "model__eval_metric": ["logloss"],
    "model__use_label_encoder": [False],
}
clf = GridSearchCV(pipe, param_grid=params, scoring="f1")
clf.fit(X_train, y_train)
y_val_pred = clf.best_estimator_.predict(X_val)
print(classification_report(y_val, y_val_pred))
# %%
# Refit with validation data
best_model = make_pipeline(
    XGBClassifier(
        **{
            key.removeprefix("model__"): value
            for key, value in clf.best_params_.items()
        }))
best_model.fit(X_train_total, y_train_total)
# %%
save_predictions(clf.best_estimator_, "svm.csv")