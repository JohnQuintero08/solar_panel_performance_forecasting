def clean_irregular_data(df):
    df_c = df.copy()
    df_c =  df_c.query('irradiation >= 0')
    df_c =  df_c.query('wind_speed >= 0')
    df_c =  df_c.query('precipitation >= 0')
    return df_c
    