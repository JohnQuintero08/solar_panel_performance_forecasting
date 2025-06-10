import pandas as pd

from insights.i_01_insights_functions import show_info, check_null_values, check_duplicates, plot_pairplot, plot_variable_hist, create_time_features, plot_scatter_trim_by_year, plot_seasonal_param, plot_seasonal_param_polar_trim, multi_lag_plot_trim, lag_plot_trim, plot_box


df = pd.read_feather('../data/preprocess_data.feather')


show_info(df)


check_null_values(df)


check_duplicates(df)


plot_variable_hist(df, 'irradiation')


new_df = create_time_features(df)


plot_scatter_trim_by_year(new_df,'irradiation',2012)


plot_scatter_trim_by_year(new_df,'precipitation',2012)


plot_scatter_trim_by_year(new_df,'wind_speed',2012)


plot_scatter_trim_by_year(new_df,'temperature',2012)


plot_seasonal_param(new_df,'irradiation', 2012, 2025)


plot_seasonal_param_polar_trim(new_df, 'irradiation', 2012, 2025)


multi_lag_plot_trim(new_df, 'irradiation', 2012, 2025)


lag_plot_trim(new_df, 'irradiation', 365)


lag_plot_trim(new_df, 'wind_speed', 365)


lag_plot_trim(new_df, 'precipitation', 365)


lag_plot_trim(new_df, 'temperature', 365)


plot_box(new_df, 'irradiation', 2000, 2025, 2012)





