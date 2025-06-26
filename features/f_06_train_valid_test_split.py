from sklearn.model_selection import train_test_split

def split_df_for_model(df):
    df_train, df_pass = train_test_split(df, 
                                         test_size=0.30, 
                                         shuffle=False
                                         )
    df_valid, df_test = train_test_split(df_pass, 
                                         test_size=0.5, 
                                         shuffle=False
                                         )
    return df_train, df_valid, df_test
