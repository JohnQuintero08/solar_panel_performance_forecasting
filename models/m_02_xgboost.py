import mlflow.xgboost
from xgboost import XGBRegressor
import mlflow

from features.f_00_features import df_f_train, df_f_valid
from models.m_01_evaluation_functions import model_evaluation_mlflow


xgboost_params = {
'objective'     :'reg:squarederror',
'eval_metric'   :'rmse',
'learning_rate' : 0.03, 
'max_depth'     : 15, 
'subsample'     : 0.8,
'colsample_bytree': 0.8,
'n_estimators'  : 100,
'alpha'         : 1,
'random_state'  : 12345}

model_xgboost = XGBRegressor(**xgboost_params)

xgb_metrics = model_evaluation_mlflow(model_xgboost, df_f_train, df_f_train,'irradiation', 'XGBoost')

mlflow.set_experiment("Time series experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
  mlflow.log_params(xgboost_params)
  mlflow.log_metrics(xgb_metrics)
  mlflow.xgboost.log_model(model_xgboost, "XGBoost")