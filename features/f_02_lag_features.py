def lag_variable(df, variable, lag):
    df_c = df.copy()
    df_c[f'{variable}_lag{lag}'] = df_c[variable].shift(lag)
    return df_c

def lag_all_features(df, lag):
    df_c = df.copy()
    features = ['irradiation', 'temperature', 'wind_speed', 'precipitation']
    for feature in features:
        df_c = lag_variable(df_c, feature, lag)
    return df_c