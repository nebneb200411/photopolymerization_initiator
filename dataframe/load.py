import pandas as pd

def load_csv(df_path):
    df = pd.read_csv(df_path, index_col=0)
    return df

def to_list(series):
    listed = series.values.tolist()
    return listed