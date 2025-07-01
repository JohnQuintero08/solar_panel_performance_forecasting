def create_time_features(df):
    df_c = df.copy()
    df_c['year'] = df_c['date'].dt.year
    df_c['month'] = df_c['date'].dt.month
    df_c = df_c.set_index('date', drop=True)
    return df_c