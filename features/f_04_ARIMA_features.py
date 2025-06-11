def arima_transform_df(df):
    df_c = df.copy()
    df_c = df_c.set_index('date', drop=True)
    df_c = df_c.drop(['temperature', 'wind_speed', 'precipitation'], axis=1)
    return df_c