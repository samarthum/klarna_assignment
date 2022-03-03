# Library Imports
from fastapi import FastAPI
from src.customer_info import CustomerInfo
import pickle
import pandas as pd
import os

ROOT_PATH = os.getcwd()
APP_PATH = os.path.join(ROOT_PATH, 'src')
MODELS_PATH = os.path.join(APP_PATH, 'models')
XGB_PATH = os.path.join(MODELS_PATH, 'xgboost.pkl')
OHE_PATH = os.path.join(MODELS_PATH, 'ohe.pkl')

# Read in all the relevant columns used for the analysis and modelling
with open(os.path.join(MODELS_PATH, 'final_columns.pkl'), 'rb') as file:
    FEATURES_USED = pickle.load(file)

with open(os.path.join(MODELS_PATH, 'cat_cols.pkl'), 'rb') as file:
    CATEGORICAL_COLUMNS = pickle.load(file)

with open(os.path.join(MODELS_PATH, 'num_cols.pkl'), 'rb') as file:
    NUMERIC_COLUMNS = pickle.load(file)

with open(os.path.join(MODELS_PATH, 'model_cols.pkl'), 'rb') as file:
    MODEL_FEATURES = pickle.load(file)

# Load the One Hot Encoder object
with open(OHE_PATH, 'rb') as file:
    ohe = pickle.load(file)

# Load the XGBoost Model
with open(XGB_PATH, 'rb') as file:
    model = pickle.load(file)


# Create the app object
app = FastAPI()


def get_ohe_data(cat_data):
    """
    Converts given categorical columns to numeric columns by performing
    One Hot Encoding.
    The encoder has been trained on the training data.

    Parameters
    ----------
    cat_data : DataFrame
        DataFrame containing the categorical features.

    Returns
    -------
    cat_data : DataFrame
        Dataframe with the features one hot encoded.

    """
    ohe_data = ohe.transform(cat_data)
    cat_data = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(),
                            index=cat_data.index).astype(int)
    return cat_data


def process_data(data):
    """
    Takes in the raw data and processes it to make it ready for the
    model prediction.

    Parameters
    ----------
    data : DataFrame
        Raw Data.

    Returns
    -------
    None.

    """
    data = dict(data)

    uuid = data['uuid']
    data = pd.DataFrame(data, index=[0])

    print(data)

    data = data[FEATURES_USED]
    cat_data = data[CATEGORICAL_COLUMNS]
    num_data = data[NUMERIC_COLUMNS]

    cat_data = get_ohe_data(cat_data)

    data = pd.concat([num_data, cat_data], axis=1)

    data = data[MODEL_FEATURES]

    return uuid, data


@app.post('/predict')
def predict_default(data: CustomerInfo):
    # Convert data to dict and filter relevant features
    uuid, data = process_data(data)
    prediction = round(float(model.predict_proba(data)[:, 1][0]), 3)

    return {'uuid': uuid, 'pd': prediction}
