import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import inspect

def encode_var(col, nunique_limit=20):
    if len(col.unique())<nunique_limit:
        one_hot_encoded = pd.get_dummies(col)
        col = one_hot_encoded.to_numpy().tolist()
    else:
        label_encoder = LabelEncoder()
        col = label_encoder.fit_transform(col)