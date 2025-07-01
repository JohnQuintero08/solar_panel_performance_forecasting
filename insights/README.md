# INSIGHTS OVERVIEW

- The first thing that is noticed is the presence of one irregular data that has an irradiance value of -999, a boudary function is created in the preprocess section to avoid incluiding this type of data.
- The dataset doesn't have null values or duplicates
- This analysis is done for the city of Valencia, Spain.
- Variables such as irradiation and temperature exhibit a clear seasonal pattern that repeats annually, as shown in the scatterplots. These variables tend to have their lowest values at the beginning and end of the year, with a peak occurring around mid-year.
- Wind speed also shows a seasonal behavior similar to irradiation and temperature, although the pattern is less pronounced at first glance.
- The precipitation variable does not display an obvious seasonal pattern, though some isolated high values are observed on certain days throughout the years.
- The lag plot analysis indicates the highest correlation for the irradiation variable occurs with a lag of 180 days.
- Autocorrelation analysis confirms this finding: there is an inverse autocorrelation at around 180 days, and the strongest positive autocorrelation at 365 days. This suggests a cyclical behavior, where autocorrelation decreases and increases again over a one-year period. A similar pattern is observed for the temperature variable.
- For precipitation and wind speed, this pattern does not apply, as no consistent lag-based correlation is found.
- Finally, monthly boxplots also reflect the previously described seasonal trend, with increasing values toward the middle months of the year. However, anomalous irradiation values are observed in some years between May and August.
- When performing the time series decomposition of the irradiation variable, several periods are analyzed. It is observed that periods shorter than 10 produce the lowest residuals. However, when examining the plot, the seasonal component does not properly capture the oscillations. Therefore, other periods are evaluated, and it is found that a period of 360 extracts the seasonal component much more effectively and also reduces the average of the residual values, reaching nearly zero.
- The STL decomposition model, compared to the classical model, performs better in capturing the full trend over longer periods.
- The irradiation shows a non stationary behaviour, because is a seasonal variable.
- The data will be simplified to the monthly average, in order t simplify the model prediction and the ARIMA model process.
