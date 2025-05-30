import pandas as pd

from preprocess.pr_01_dataset_construction import load_json_data

def format_columns(df):
    df_c = df.copy()
    df_c['date'] =  pd.to_datetime(df['date'])
    return df_c


# Testing function
# location = 'data/initial_data.json'
# df = load_json_data(location)
# df_c = format_columns(df)

# df_c.to_feather('data/preprocess_data.feather')
