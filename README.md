# Credit Default Risk Prediction

This project predicts the probability of credit card default using TabNet and several baseline machine learning models.

Dataset: Default of Credit Card Clients (UCI Machine Learning Repository)
Each record contains demographic and financial information about a borrower.
The target variable is `default_payment_next_month` (1 = default, 0 = non-default).

## 1. Project structure:

tabnet_application/
  default-of-credit-card-clients-predictive-models.ipynb   - Jupyter notebook for training and evaluation
  default_of_credit_card_clients.csv                       - Dataset file
  tabnet_credit_default.zip                                - Trained model file
  catboost_info/                                           - CatBoost logs and auxiliary data
  README.md                                                - This documentation

Models used:
  - Logistic Regression
  - Random Forest
  - XGBoost / CatBoost
  - TabNet (main model)

TabNet combines tree-like feature selection with neural network learning, designed for tabular structured data.

## 2. Main workflow:
1. Data preprocessing
   - Handle missing values
   - Normalize numeric features
   - Encode categorical variables

2. Model training
   - Split dataset into training and test sets
   - Train models, tune hyperparameters
   - Evaluate using standard classification metrics

3. Model evaluation
   - Compare baseline models with TabNet
   - Analyze feature importance and interpretability

## 3. Requirements:
  pip install pandas numpy torch pytorch-tabnet scikit-learn matplotlib

## 4. Usage:
### Prepare the environment
Make sure Python 3.8 or above is installed.

Install required libraries:
pip install pandas numpy torch pytorch-tabnet scikit-learn matplotlib


If you want to reproduce baseline models such as XGBoost or CatBoost:
pip install xgboost catboost

### Prepare the dataset
Download or ensure the dataset file `default_of_credit_card_clients.csv` is located in the same folder as the notebook.
This dataset should have columns such as:
- `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- `PAY_0` to `PAY_6`
- `BILL_AMT1` to `BILL_AMT6`
- `PAY_AMT1` to `PAY_AMT6`
- `default_payment_next_month` (target)

No manual cleaning is needed if you use the provided notebook.

In the real world banking. They usually allow less FPR, to ensure customer experience. user can change 'alpha' in the neyman-pearson to see how FPR and recall changes on different cut-off.
In order to use Neyman-Pearson Hypothesis testing running Neyman_Pearson.py: 

```python
  dt_recall = run_np_analysis(best_model, "Decision Tree", val_df[predictors], val_df[target].values)
```

### Run the notebook

## 5. Author:
Yankai Zhao

