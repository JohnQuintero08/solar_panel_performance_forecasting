# Test Augmented Dickey-Fuller (ADF)
# Hipostesis nula: La serie no es estacionaria
# p-value < 0.05 → Rechazas la hipótesis nula → La serie es estacionaria

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(df):
    result = adfuller(df)
    # Mostrar resultados
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1]< 0.05:
        print('The time series is statioanry')
    else:
        print('The time series is non stationary')
        