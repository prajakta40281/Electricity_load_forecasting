# All-India Electricity Load Forecasting

Hourly national electricity demand forecasting using ML models, benchmarked against naive baselines.

## Data
- Hourly all-India demand, Jan 2019 – Jun 2025.
- Temperature (avg/max/min) and rainfall. 
- GDP, mapped by fiscal year. Source: NITI Aayog
- File: `dataset/india_electricity_load_dataset_FINAL.csv`

## Method
- Features: lag 1h / 24h / 168h, sin/cos encodings of hour, day-of-week and month, avg temperature, rainfall, GDP
- Chronological split: train ≤ Dec 2024, test Jan–Jun 2025.
- Hyperparameter tuning with `TimeSeriesSplit`
- Models: MLR, Random Forest, SVR, AdaBoost, Gradient Boosting, XGBoost
- Baselines: last hour, same hour yesterday, same hour last week

## Results (one-hour-ahead, test Jan–Jun 2025)

| Model | MAE | MAPE | RMSE | R² |
|---|---|---|---|---|
| **Gradient Boosting (tuned)** | **2284** | **1.16%** | **3813** | **0.962** |
| XGBoost (tuned) | 2322 | 1.18% | 3851 | 0.961 |
| AdaBoost (tuned) | 2616 | 1.32% | 4095 | 0.956 |
| Random Forest | 2729 | 1.38% | 4158 | 0.955 |
| MLR | 3479 | 1.78% | 5038 | 0.934 |
| SVR (tuned) | 4367 | 2.15% | 6511 | 0.889 |
| Naive: last hour | 4599 | 2.36% | 6449 | 0.891 |
| Naive: same hour yesterday | 5950 | 3.07% | 9075 | 0.785 |
| Naive: same hour last week | 9012 | 4.58% | 12537 | 0.589 |

Boosting methods perform best. 

## Run
```
pip install pandas numpy scikit-learn xgboost matplotlib jupyter
jupyter notebook load_forecasting.ipynb
```
