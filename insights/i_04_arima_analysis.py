import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from insights.i_03_stationary_test import adf_test

def plot_acf_pacf(df, lag):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(df, lags=lag, ax=axs[0])
    axs[0].set_title("ACF")

    plot_pacf(df, lags=lag, ax=axs[1], method='ywm')
    axs[1].set_title("PACF")

    plt.tight_layout()
    plt.show()

def arima_transform_df(df):
    df_c = df.copy()
    df_c = df_c.set_index('date', drop=True)
    df_c = df_c.drop(['temperature', 'wind_speed', 'precipitation'], axis=1)
    df_c = df_c.asfreq('D', method='ffill')
    df_c = df_c.resample('ME').mean()
    return df_c


df = pd.read_feather('data/preprocess_data.feather')
df_arima = arima_transform_df(df)
df_arima =df_arima.diff(12).dropna()
adf_test(df_arima)

plot_acf_pacf(df_arima,40)

df_arima.plot(style='--')
plt.show()
