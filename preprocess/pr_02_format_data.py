import pandas as pd

from preprocess.pr_01_dataset_construction import load_json_data

def format_columns(df):
    df_c = df.copy()
    df_c['date'] =  pd.to_datetime(df['date'])
    return df_c
