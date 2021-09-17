# %%
from numpy.lib.npyio import save
from utils import make_pipeline, X_train, X_val, save_predictions, y_train, y_val, df_test
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# %%
pipe = make_pipeline(SVC())
# %%
pipe.fit(X_train, y_train)
y_val_pred = pipe.predict(X_val)
# %%
print(classification_report(y_val, y_val_pred))
# %%
params = {
    "model__penalty": ["l1", "l2"],
    "model__alpha": [0.01, 0.1, 0.001, 0.0001],
    "model__loss":
    ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    "model__eta0": [10, 1e-2, 1e-3, 1e-4],
    "model__learning_rate": ['optimal', 'constant', 'adaptive']
}
clf = GridSearchCV(pipe, param_grid=params, scoring="f1")
clf.fit(X_train, y_train)
y_val_pred = clf.best_estimator_.predict(X_val)
print(classification_report(y_val, y_val_pred))
# %%
save_predictions(clf.best_estimator_, "sgd.csv")