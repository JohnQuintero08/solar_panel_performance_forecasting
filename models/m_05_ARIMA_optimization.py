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
