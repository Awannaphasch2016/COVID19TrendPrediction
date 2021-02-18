# DATASET
* reference:
    * us-state.csv (state-level data)
        * https://github.com/nytimes/covid-19-data
# BASELINE MODELS 
* stochastic (random walk) basline
    * previous_day_prediciton
* basic baseline
    * Autoregressive Integrated Moving Average (ARIMA)
    * Linear Regression
* state of the art machine learning 
    * XGBoost (gradient boosting tree.)
* statistical baseline model
    * GAM
* Deep learning baseline model
    * MLP
* Deep learning time series baslines model
    * LSTM

# EVALUATION METRICS
    * mape
    * mse
    * rmse
    * r2score

# Valiadtion process
* expanding window validation (train/val/test)
    * aka walking forward validation
