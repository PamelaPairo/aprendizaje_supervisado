# %%
from utils import make_pipeline, X_train, X_val, y_train, y_val
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# %% [markdown]
# ## Default
# %%
seed = 0
pipe = make_pipeline(LogisticRegression(random_state=seed))
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)
# %%
text = "Logistic Regression - Reporte de clasificación del conjunto de validation"
print(len(text) * "=")
print(text)
print(len(text) * "=")
print(classification_report(y_val, y_val_pred))
# %% [markdown]
# ## Fine Tunning
# %%
params = {
    "model__penalty": ["l1", "l2"],
    "model__class_weight": ["balanced"],
    "model__solver": ["liblinear", "saga"],
    "model__max_iter": [5000]
}
clf = GridSearchCV(pipe, param_grid=params, scoring="f1")
clf.fit(X_train, y_train)
y_val_pred = clf.best_estimator_.predict(X_val)
print(len(text)*"=")
print(text)
print(len(text)*"=")
print(classification_report(y_val, y_val_pred))
# %%
