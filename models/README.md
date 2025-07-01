# MODELS OVERVIEW

- The XGBoost and SARIMA models were used to train and generate predictions for the time series.
- A SARIMA model was used because the data has a significant seasonal component.
- The models were tested by changing the hyperparameters, and the experiments were tracked using MLflow. Additionally, the experiments were stored in an external Dagshub repository, which can be accessed through this link:  
  [https://dagshub.com/JohnQuintero08/solar_panel_performance_forecasting](https://dagshub.com/JohnQuintero08/solar_panel_performance_forecasting)
- After exploring different hyperparameters, a range was defined to optimize the models. The optimization was done using scikit-learn's `GridSearch` for the XGBoost model and `AutoARIMA` for the SARIMA model.

---

### Optimized Results

#### Model Evaluation: XGBoost

- Dataset – Train
  - RMSE: `0.07`
  - MAE: `0.04`
- Dataset – Validation
  - RMSE: `0.08`
  - MAE: `0.05`

#### Model Evaluation: SARIMA

- Dataset – Validation
  - RMSE: `0.36`
  - MAE: `0.27`

---

- The XGBoost model shows better results than the SARIMA model and fits the validation data more effectively without being overfitted, compared to SARIMA's validation performance.
- The XGBoost model reduces the RMSE by a factor of 4 compared to the RMSE produced by the SARIMA model.
- Finally the model XGBoost with the test dataset performed well and the results were:

- Dataset – Test
  - RMSE: `0.12`
  - MAE: `0.07`
