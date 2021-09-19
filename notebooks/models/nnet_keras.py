# %%
import pandas as pd
from utils import (make_pipeline, df_test,  X_train_total, y_train_total)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# %%
def create_model(hl1=100,
                 hl2=50,
                 activation="relu",
                 optimizer="adam",
                 dropout=.2,
                 kernel_initializer="normal"):
    model = Sequential()
    model.add(
        Dense(hl1,
              kernel_initializer=kernel_initializer,
              activation=activation))
    model.add(Dropout(dropout))
    model.add(
        Dense(hl2,
              kernel_initializer=kernel_initializer,
              activation=activation))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model
# %%
pipe = make_pipeline(KerasClassifier(build_fn=create_model, verbose=0))

params_bayes = {
    "preprocessor__discretizer__n_bins": Integer(2, 25),
    "model__batch_size": Integer(10, 100),
    "model__epochs": Integer(10, 50),
    "model__hl1": Integer(50, 200),
    "model__hl2": Integer(25, 100),
    "model__activation": Categorical(["relu", "tanh"]),
    "model__optimizer": Categorical(["adam", "sgd"]),
    "model__dropout": Real(0, .4),
    "model__kernel_initializer": Categorical(["uniform", "normal"])
}
# %%
opt = BayesSearchCV(pipe, params_bayes, scoring="f1")
opt.fit(X_train_total, y_train_total)
# %%
# %%
test_id = df_test["Customer"]
test_pred = opt.best_estimator_.predict(df_test.drop(columns=["Customer"]))
submission = pd.DataFrame(list(zip(test_id, test_pred)),
                              columns=["Customer", "TravelInsurance"])
submission["TravelInsurance"] = submission.TravelInsurance.apply(lambda xs: xs[0])
# %%
