import numpy as np
import pandas as pd

import pickle

ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler_model = pickle.load(open("models/scaler.pkl", "rb"))


def fwi_prediction(temperature, rh, ws, rain, ffmc, dmc, isi, classes, region):

    new_scaled_data = scaler_model.transform(
        [[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]
    )
    result = ridge_model.predict(new_scaled_data)
    return result
