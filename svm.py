# %%
from utils import (make_pipeline, X_train, X_val, save_predictions, y_train,
                   y_val, X_train_total, y_train_total)
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# %% [markdown]
# ## Default
# %%
pipe = make_pipeline(SVC())
# %%
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)
# %%
print(classification_report(y_val, y_val_pred))
# %% [markdown]
# ## Fine Tunning
# %%
params = {
    "model__kernel": ["linear", "sigmoid", "poly", "rbf"],
    "model__gamma": ["scale", "auto"],
    "model__degree": [2, 3, 4],
    "model__coef0": [.001, .01, 0, 1],
    "model__tol": [1e-2, 1e-3, 1e-4],
    "model__C": [1, 0.1, 0.01, 0.001, 0.0001, 10],
    "model__class_weight": ["balanced"]
}
clf = GridSearchCV(pipe, param_grid=params, scoring="f1")
clf.fit(X_train, y_train)
y_val_pred = clf.best_estimator_.predict(X_val)
print(classification_report(y_val, y_val_pred))
# %%
# Refit with validation data
best_model = make_pipeline(
    SVC(
        **{
            key.removeprefix("model__"): value
            for key, value in clf.best_params_.items()
        }))
best_model.fit(X_train_total, y_train_total)
# %%
save_predictions(clf.best_estimator_, "svm.csv")
# %%
