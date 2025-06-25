import mlflow
import mlflow.xgboost
import mlflow.data
from mlflow.models import infer_signature
import mlflow.xgboost
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor
import dagshub
import os
from dotenv import load_dotenv

from features.f_00_features import df_f_train, df_f_valid
from models.m_01_evaluation_functions import model_evaluation_mlflow
from features.f_05_features_target_split import features_target_split

load_dotenv()

mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
dagshub.init(repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name='solar_panel_performance_forecasting', mlflow=True)

def run_experiment(registration=False):
  random_id_experiment = np.random.randint(0, 10000)
  with mlflow.start_run(run_name= f'XGBoost-{random_id_experiment}'):
    # Define model params in a dictionary
    xgboost_params = {
    'objective'     :'reg:squarederror',
    'eval_metric'   :'rmse',
    'learning_rate' : 0.05, 
    'max_depth'     : 8, 
    'subsample'     : 0.8,
    'colsample_bytree': 1,
    'n_estimators'  : 100,
    'alpha'         : 1,
    'random_state'  : 12345}
    # Create the model
    model_xgboost = XGBRegressor(**xgboost_params)
    # Evaluate the model comparing train and validation
    xgb_metrics = model_evaluation_mlflow(model_xgboost, df_f_train, df_f_valid, 'irradiation', 'XGBoost', False)
    # Log the datasets
    ds_train = mlflow.data.from_pandas(df_f_train, name='dataset with features', targets='irradiation')
    ds_val = mlflow.data.from_pandas(df_f_valid, name='dataset with features', targets='irradiation')
    mlflow.log_input(ds_train, context="training")
    mlflow.log_input(ds_val, context="validation")
    # Log params and metrics
    mlflow.log_params(xgboost_params)
    mlflow.log_param('model_name', f'XGBoost-{random_id_experiment}')
    mlflow.log_metrics(xgb_metrics)
    
    if registration:
      mlflow.xgboost.log_model(model_xgboost, name="xgboost",)

run_experiment()