from xgboost import XGBRegressor

from features.f_00_features import df_f_train, df_f_valid
from models.m_01_evaluation_functions import model_evaluation

ran=12345

model_xgboost = XGBRegressor(objective='reg:squarederror',
                              eval_metric='rmse',
                                learning_rate = 0.03, 
                                max_depth=15, 
                                subsample=0.8,
                                colsample_bytree=0.8,
                                n_estimators=100,
                                alpha=1,
                                random_state=ran)


model_evaluation(model_xgboost, df_f_train, df_f_valid,'irradiation', 'XGBoost', True)