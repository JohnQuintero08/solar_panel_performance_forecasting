def moving_average_variable(df, feature, period):
    df_c = df.copy()
    df_c[f'rolling_{feature}_{period}p'] = df_c[feature].rolling(window=period).mean()
    return df_c

def ma_all_features(df, period):
    df_c = df.copy()
    features = ['irradiation', 'temperature', 'wind_speed', 'precipitation']
    for feature in features:
        df_c = moving_average_variable(df_c, feature, period)
    return df_c