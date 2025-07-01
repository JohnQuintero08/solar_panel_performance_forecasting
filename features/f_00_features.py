import pandas as pd

from features.f_04_ARIMA_features import arima_transform_df
from features.f_08_tree_boost_features import create_features_df, create_selected_features_df
from features.f_06_train_valid_test_split import split_df_for_model

df = pd.read_feather('data/preprocess_data.feather')


# FEATURES DATASET

df_f = create_features_df(df)
df_f_train, df_f_valid, df_f_test = split_df_for_model(df_f)

# FEATURES DATASET CHOPED
df_f_selected = create_selected_features_df(df)


#  ARIMA DATASET

df_arima = arima_transform_df(df)
df_arima_train, df_arima_valid, df_arima_test = split_df_for_model(df_arima)
