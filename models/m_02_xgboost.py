import dagshub
from dotenv import load_dotenv
import mlflow
import mlflow.xgboost
import mlflow.data
from mlflow.models import infer_signature
import numpy as np
import os
import yaml
from xgboost import XGBRegressor

from features.f_00_features import df_f
from models.m_01_evaluation_functions import model_evaluation_mlflow
from features.f_05_features_target_split import features_target_split
from features.f_06_train_valid_test_split import split_df_for_model

load_dotenv()
mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
dagshub.init(repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), repo_name='solar_panel_performance_forecasting', mlflow=True)


def run_experiment(is_test=False, registration=False):
  
  # Split the dataset
  df_f_train, df_f_valid, df_f_test = split_df_for_model(df_f)
  # Creates a random ID identification for the run
  random_id_experiment = np.random.randint(0, 10000)
  
  with mlflow.start_run(run_name= f'XGBoost-{'Test-' if is_test else ''}{random_id_experiment}'):
    # Open and check the DVC hash of the file
    with open("data/preprocess_data.feather.dvc", "r") as f:
      dvc_data = yaml.safe_load(f)
      dvc_hash = dvc_data["outs"][0]["md5"]
      
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
    xgb_metrics = model_evaluation_mlflow(model = model_xgboost, 
                                          df_train = df_f_train, 
                                          df_pr = df_f_test if is_test else df_f_valid, 
                                          target_name = 'irradiation', 
                                          model_name = 'XGBoost', 
                                          has_plot = True,
                                          df_name_1="train",
                                          df_name_2="test" if is_test else "valid")
    
    # Log the original datasets that is controled with DVC
    ds = mlflow.data.from_pandas(df_f, 
                                  name='preprocess dataset', 
                                  targets='irradiation',
                                  digest=dvc_hash) # Link the DVC hash with the experiment
    mlflow.log_input(ds, context="origin")

    ds_train = mlflow.data.from_pandas(df_f_train, name='dataset with features', targets='irradiation')
    mlflow.log_input(ds_train, context="training")
    
    if is_test:
      ds_test = mlflow.data.from_pandas(df_f_test, name='dataset with features', targets='irradiation')
      mlflow.log_input(ds_test, context="test")
    else:
      ds_val = mlflow.data.from_pandas(df_f_valid, name='dataset with features', targets='irradiation')
      mlflow.log_input(ds_val, context="validation")
        
    
    # Log params and metrics
    mlflow.log_params(xgboost_params)
    mlflow.log_param('model_name', f'XGBoost-{random_id_experiment}')
    mlflow.log_metrics(xgb_metrics)
    
    if registration:
      f_t, t_t = features_target_split(df_f_train.iloc[0:1,:], 'irradiation')
      signature = infer_signature(f_t, t_t)
      mlflow.xgboost.log_model(model_xgboost, artifact_path="xgboost", signature=signature)

run_experiment(is_test=False, registration=False)