from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.m_01_evaluation_functions import metrics_eval, graph_predictions

from features.f_00_features import df_arima_train, df_arima_valid

# ARIMA(0, 0, 1)x(0, 1, 2, 12)12 - AIC:142.87765088747216
model = SARIMAX(df_arima_train,
                order=(1,1,0),
                seasonal_order=(2,1,1,12))

results = model.fit()

forecast = results.forecast(steps=45)  # 12 meses siguientes
forecast.plot(label='Pronóstico', color='orange')
df_arima_valid.plot(label='Real_value')
df_arima_train.plot(label='Datos históricos', figsize=(10,4))
plt.legend()

plt.show()

def arima_model_evaluation(model, df_to_predict, step, model_name, has_plot):
    results = model.fit()
    predictions = results.forecast(steps= step)
    print(f"Model evaluation: {model_name}")
    metrics_eval(df_to_predict, predictions, 'Validation')
    
    if has_plot:
        plt.scatter(df_to_predict, predictions, s=20, alpha=0.8)
        plt.plot([df_to_predict.min(), df_to_predict.max()], [df_to_predict.min(), df_to_predict.max()], 'r--')
        plt.title("SARIMA predictions for irradiation")
        plt.xlabel('Real value')
        plt.ylabel('Prediction')
        plt.legend(['Predictions', 'Ideal correlation'])
        plt.tight_layout()
        plt.show()

    
arima_model_evaluation(model, df_arima_valid, 45, 'SARIMA', True)




import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Define the p, d, q ranges
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Initialize lists to store the parameters and AIC values
list_param = []
list_param_seasonal = []
list_results_aic = []

# Initialize variables to track the best model
best_aic = float("inf")
best_param = None
best_param_seasonal = None

# Loop through all combinations of pdq and seasonal_pdq
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            # Fit the SARIMA model
            model = SARIMAX(df_arima_train,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)

            results = model.fit()

            # Append parameters and AIC to the lists
            list_param.append(param)
            list_param_seasonal.append(param_seasonal)
            list_results_aic.append(results.aic)

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            # Check if this model has the lowest AIC
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
                best_param_seasonal = param_seasonal

        except:
            continue

# Print the best model parameters
print("\nBest Model:")
print('ARIMA{}x{}12 - AIC:{}'.format(best_param, best_param_seasonal, best_aic))