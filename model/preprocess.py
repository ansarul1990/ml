

import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_features_target(df: pd.DataFrame, target_col):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x,y

def scale_height_weight(x:pd.DataFrame):
    scaler = StandardScaler()
    if "Height" in x.columns and "Weight" in x.columns:
        x = x.copy()
        x[["Height", "Weight"]] = scaler.fit_transform((x[["Height", "Weight"]]))
    return x


