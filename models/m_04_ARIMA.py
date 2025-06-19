import mlflow
import mlflow.data
import mlflow.statsmodels
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from features.f_00_features import df_arima_train, df_arima_valid
from models.m_01_evaluation_functions import arima_model_evaluation_mlflow

mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def run_experiment():

    random_id_experiment = np.random.randint(0, 10000)
    
    with mlflow.start_run(run_name= f'SARIMA-{random_id_experiment}'):
        # Define model params in a dictionary
        # ARIMA(0, 0, 1)x(0, 1, 2, 12)12 - AIC:142.87765088747216
        sarima_params = {
            'order' : (0,0,1),
            'seasonal_order' : (0,1,2,12),
            'enforce_stationarity': True,
            'enforce_invertibility': True
        }
        # Create the model
        model_sarima = SARIMAX(endog=df_arima_train,**sarima_params)
        # Evaluate the model comparing train and validation
        sarima_metrics = arima_model_evaluation_mlflow(model_sarima, df_arima_valid, 45, 'SARIMA', True)
        # Log the datasets
        ds_train = mlflow.data.from_pandas(df_arima_train, name='dataset arima', targets='irradiation')
        ds_val = mlflow.data.from_pandas(df_arima_valid, name='dataset arima', targets='irradiation')
        mlflow.log_input(ds_train, context="training")
        mlflow.log_input(ds_val, context="validation")
        # Log params and metrics
        mlflow.log_params(sarima_params)
        mlflow.log_param('model_name', f'SARIMA-{random_id_experiment}')
        mlflow.log_metrics(sarima_metrics)
        # Log and save the model in MLFlow with an input example
        # mlflow.statsmodels.log_model(model_sarima, "SARIMA")
    
run_experiment()
