from features.f_07_resample_data import resample_data

def arima_transform_df(df):
    df_c = df.copy()
    df_c = df_c.set_index('date', drop=True)
    df_c = df_c.drop(['temperature', 'wind_speed', 'precipitation'], axis=1)
    df_c = resample_data(df_c, 'ME')
    return df_c