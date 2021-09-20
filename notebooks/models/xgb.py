# %%
from utils import (make_pipeline, save_predictions, X_train_total,
                   y_train_total)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# %%
pipe = make_pipeline(XGBClassifier())
# %% [markdown]
# ## Fine Tunning
# %%
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
opt = BayesSearchCV(pipe, params_bayes, scoring="f1")
opt.fit(X_train_total, y_train_total)
# %%
save_predictions(opt.best_estimator_, "xgb.csv")
# %%
