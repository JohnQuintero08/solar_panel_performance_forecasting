from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.m_01_evaluation_functions import metrics_eval, graph_predictions

from features.f_00_features import df_arima_train, df_arima_valid
from models.m_01_evaluation_functions import arima_model_evaluation

# ARIMA(0, 0, 1)x(0, 1, 2, 12)12 - AIC:142.87765088747216
model = SARIMAX(df_arima_train,
                order=(1,1,0),
                seasonal_order=(2,1,1,12))

results = model.fit()


arima_model_evaluation(model, df_arima_valid, 45, 'SARIMA', True)
