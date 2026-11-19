# Model Performance Comparison

Evaluation performed on California Housing Dataset (80:20 Train-Test split, random_state=42):

| Algorithm | MAE | MSE | RMSE | R² Score | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Linear Regression | 0.5332 | 0.5559 | 0.7456 | 0.5758 | 0.07 |
| Decision Tree | 0.4171 | 0.3953 | 0.6287 | 0.6983 | 0.25 |
| Random Forest | 0.3277 | 0.2558 | 0.5057 | 0.8048 | 7.34 |
| Neural Network | 0.3533 | 0.2745 | 0.5239 | 0.7905 | 9.88 |
| XGBoost | 0.3091 | 0.2163 | 0.4650 | 0.8350 | 0.77 |
| KNN | 0.4375 | 0.4191 | 0.6474 | 0.6801 | 0.07 |

**Best Performing Algorithm:** `XGBoost` with R² Score of `0.8350` and lowest RMSE of `0.4650`.
