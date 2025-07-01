from features.f_01_time_features import create_time_features
from features.f_02_lag_features import lag_all_features, lag_variable
from features.f_03_moving_average_features import ma_all_features
from features.f_07_resample_data import resample_data

def create_features_df(df):
    new_df = df.copy()
    new_df = create_time_features(df)
    new_df = resample_data(new_df, 'ME')
    periods = [1,3,6,12]
    for period in periods:
        new_df = lag_all_features(new_df, period)
        new_df = ma_all_features(new_df, period)
    new_df = new_df.dropna(axis=0)
    return new_df

def create_selected_features_df(df):
    new_df = df.copy()
    new_df = create_time_features(df)
    new_df = resample_data(new_df, 'ME')
    new_df = lag_variable(new_df, 'irradiation', 1)
    new_df = lag_variable(new_df, 'precipitation', 1)
    new_df = lag_variable(new_df, 'temperature', 3)
    new_df = new_df.dropna(axis=0)
    new_df = new_df.drop(['temperature','wind_speed','precipitation','year','month'], axis=1)
    return new_df
