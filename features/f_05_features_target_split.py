def features_target_split(df, target):
    features = df.drop([target], axis=1)
    target = df[target]
    return features, target