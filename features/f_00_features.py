import pandas as pd

from features.f_01_time_features import create_time_features
from features.f_02_lag_features import lag_all_features
from features.f_03_moving_average_features import ma_all_features
from features.f_04_ARIMA_features import arima_transform_df
from features.f_06_train_valid_test_split import split_df_for_model

def create_features_df(df):
    new_df = df.copy()
    new_df = create_time_features(df)
    
    periods = [1,7,30,180,360]
    for period in periods:
        new_df = lag_all_features(new_df, period)
        new_df = ma_all_features(new_df, period)
    new_df = new_df.dropna(axis=0)
    return new_df

df = pd.read_feather('data/preprocess_data.feather')


# FEATURES DATASET

df_f = create_features_df(df)
df_f_train, df_f_valid, df_f_test = split_df_for_model(df_f)

#  ARIMA DATASET

df_arima = arima_transform_df(df)
df_arima_train, df_arima_valid, df_arima_test = split_df_for_model(df_arima)
