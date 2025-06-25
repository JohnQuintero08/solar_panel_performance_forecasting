import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from features.f_00_features import df_f_train, df_f_valid
from features.f_05_features_target_split import features_target_split

df_cross_validation = pd.concat([df_f_train, df_f_valid], axis=0)
features_cross_validation, target_cross_validation = features_target_split(df_cross_validation, 'irradiation')

xgboost_base_params = {
    'objective'     :'reg:squarederror',
    'eval_metric'   :'rmse',
    'random_state'  : 12345
}

model_xgboost = XGBRegressor(**xgboost_base_params)

xgboost_params = {
    'learning_rate' : [0.01,0.03,0.05], 
    'max_depth'     : [6, 10, 15],
    'subsample'     : [0.6, 0.8, 1],
    'colsample_bytree':[0.6, 0.8, 1],
    'n_estimators'  : [50,70,100],
}
grid_search = GridSearchCV(estimator=model_xgboost,
                           param_grid=xgboost_params,
                           verbose=3,
                           scoring= 'neg_root_mean_squared_error', 
                           cv=3,)

grid_search.fit(features_cross_validation, target_cross_validation)


best_xgb = grid_search.best_estimator_


grid_search.best_params_
# {'colsample_bytree': 1, 'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 100, 'subsample': 0.8}