import dagshub
from dotenv import load_dotenv
import mlflow
import mlflow.data
import mlflow.statsmodels
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml

from features.f_00_features import df_arima
from models.m_01_evaluation_functions import arima_model_evaluation_mlflow
from features.f_06_train_valid_test_split import split_df_for_model

load_dotenv()
mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
dagshub.init(repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name='solar_panel_performance_forecasting', mlflow=True)

def run_experiment(registration = False):
    
    df_arima_train, df_arima_valid, df_arima_test = split_df_for_model(df_arima)

    random_id_experiment = np.random.randint(0, 10000)
    
    with mlflow.start_run(run_name= f'SARIMA-{random_id_experiment}'):
        # Open and check the DVC hash of the file
        with open("data/preprocess_data.feather.dvc", "r") as f:
            dvc_data = yaml.safe_load(f)
            dvc_hash = dvc_data["outs"][0]["md5"]
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
        model_sarima_trained, sarima_metrics = arima_model_evaluation_mlflow(model_sarima, df_arima_valid, 45, 'SARIMA', False)
        # Log the datasets
        ds_train = mlflow.data.from_pandas(df_arima_train, name='dataset arima', targets='irradiation')
        ds_val = mlflow.data.from_pandas(df_arima_valid, name='dataset arima', targets='irradiation')
        ds = mlflow.data.from_pandas(df_arima, 
                                name='dataset preprocess', 
                                targets='irradiation',
                                digest=dvc_hash) # Link the DVC hash with the experiment

        mlflow.log_input(ds, context="origin")
        mlflow.log_input(ds_train, context="training")
        mlflow.log_input(ds_val, context="validation")
        # Log params and metrics
        mlflow.log_params(sarima_params)
        mlflow.log_param('model_name', f'SARIMA-{random_id_experiment}')
        mlflow.log_metrics(sarima_metrics)
        # Log and save the model in MLFlow with an input example
        if registration:
            mlflow.statsmodels.log_model(model_sarima_trained, artifact_path="sarima")
    
run_experiment()
