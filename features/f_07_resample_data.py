def resample_data(df, resample_period):
    df_c = df.copy()
    df_c = df_c.resample(resample_period).mean()
    return df_c