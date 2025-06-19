import mlflow
import mlflow.xgboost
import mlflow.data
from mlflow.models import infer_signature
import numpy as np
from xgboost import XGBRegressor

from features.f_00_features import df_f_train, df_f_valid
from models.m_01_evaluation_functions import model_evaluation_mlflow
from features.f_05_features_target_split import features_target_split

mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def run_experiment():
  random_id_experiment = np.random.randint(0, 10000)
  with mlflow.start_run(run_name= f'XGBoost-{random_id_experiment}'):
    # Define model params in a dictionary
    xgboost_params = {
    'objective'     :'reg:squarederror',
    'eval_metric'   :'rmse',
    'learning_rate' : 0.04, 
    'max_depth'     : 20, 
    'subsample'     : 0.8,
    'colsample_bytree': 0.8,
    'n_estimators'  : 200,
    'alpha'         : 1,
    'random_state'  : 12345}
    # Create the model
    model_xgboost = XGBRegressor(**xgboost_params)
    # Evaluate the model comparing train and validation
    xgb_metrics = model_evaluation_mlflow(model_xgboost, df_f_train, df_f_valid, 'irradiation', 'XGBoost', True)
    # Log the datasets
    ds_train = mlflow.data.from_pandas(df_f_train, name='dataset with features', targets='irradiation')
    ds_val = mlflow.data.from_pandas(df_f_valid, name='dataset with features', targets='irradiation')
    mlflow.log_input(ds_train, context="training")
    mlflow.log_input(ds_val, context="validation")
    # Log params and metrics
    mlflow.log_params(xgboost_params)
    mlflow.log_param('model_name', f'XGBoost-{random_id_experiment}')
    mlflow.log_metrics(xgb_metrics)
    # Log and save the model in MLFlow with an input example and signature
    # f_t, t_t = features_target_split(df_f_train.iloc[0:1,:], 'irradiation')
    # signature = infer_signature(f_t, t_t)
    # mlflow.xgboost.log_model(model_xgboost, "XGBoost", signature=signature)
    
run_experiment()