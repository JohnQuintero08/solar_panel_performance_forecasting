from pmdarima import auto_arima

from features.f_00_features import df_arima_train, df_arima_valid

# To run this code install numpy==1.26.4 and pmdarima==2.0.4, this versions are in conflict with the current venv, that's why it's necessary to create a new venv to run this.

# The results of the optimization are: ARIMA(1,0,0)(2,0,1)[12]

stepwise_model = auto_arima(df_arima_train,
                            start_p=0, max_p=3,
                            start_q=0, max_q=3,
                            d=None,            # Autoidentify 'd'
                            seasonal=True,     # True if it's seasonal
                            m=12,              # Seasonality (12 = monthly anual cycle)
                            start_P=0, max_P=3,
                            start_Q=0, max_Q=3,
                            D=None,            # Autoidentify 'D'
                            trace=True,        # Progress
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)     # Fastest algorithm


print(stepwise_model.summary())
print(stepwise_model.order)
print(stepwise_model.seasonal_order)



# SARIMA optimization manual




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