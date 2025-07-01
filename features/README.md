# FEATURES OVERVIEW

- Two parallel models will be built to forecast the irradiation value.
- For the XGBoost model, date decomposition will be performed, and moving averages and lag features will be created for predicting the value. Additionally, the values of other variables will be used as complementary features to assess their influence.
- For the SARIMA model, since it is autoregressive, only the irradiation values will be extracted.
- Functions are also created to split the dataset into training, validation, and test sets with proportions of 70%, 15%, and 15% respectively.
- The data is adapted to monthly averages in order to reduce the computational load of the SARIMA model and improve its interpretability.
