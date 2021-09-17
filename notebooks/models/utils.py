import pandas as pd
from os import makedirs
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.model_selection import train_test_split

URL_TRAIN_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_train.csv"
URL_TEST_DATA = "https://www.famaf.unc.edu.ar/~nocampo043/travel_insurance_prediction_test.csv"
DIR_SUBMISSIONS = "submissions"

df_train = pd.read_csv(URL_TRAIN_DATA)
df_test = pd.read_csv(URL_TEST_DATA)

seed = 0

X_train_total = df_train.drop(["Customer","TravelInsurance"], axis=1)
y_train_total = df_train["TravelInsurance"]

X_train, X_val, y_train, y_val = train_test_split(X_train_total,
                                                  y_train_total,
                                                  test_size=0.2,
                                                  random_state=seed)
X_test = df_test.drop("Customer", axis=1)

preprocessor = ColumnTransformer(
    [("discretizer",
      KBinsDiscretizer(n_bins=5, encode="ordinal",
                       strategy="quantile"), ["Age", "AnnualIncome"]),
     ("encoder",
      OneHotEncoder(categories="auto", dtype="int", handle_unknown="ignore"), [
          "Employment Type", "GraduateOrNot", "FamilyMembers", "FrequentFlyer",
          "EverTravelledAbroad"
      ])],
    remainder="passthrough")

make_pipeline = lambda model: Pipeline(
    [("preprocessor", preprocessor),
     ("model", model)])

def save_predictions(model, filename):
    test_id = df_test["Customer"]
    test_pred = model.predict(df_test.drop(columns=["Customer"]))

    submission = pd.DataFrame(list(zip(test_id, test_pred)),
                              columns=["Customer", "TravelInsurance"])
    
    makedirs(DIR_SUBMISSIONS, exist_ok=True)
    submission.to_csv(f"{DIR_SUBMISSIONS}/{filename}",
                      header=True,
                      index=False)
