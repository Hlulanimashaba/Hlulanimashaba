import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib

print("Loading core & demo data...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
demo = pd.read_parquet('demographics_clean')

print("Loading and aggregating financials...")
try:
    fin = pd.read_parquet('financials_features.parquet')
    fin_features = fin.groupby('UniqueID').agg(
        fin_mean_NII=('NetInterestIncome', 'mean'),
        fin_sum_NII=('NetInterestIncome', 'sum'),
        fin_mean_NIR=('NetInterestRevenue', 'mean'),
    ).reset_index()
except Exception as e:
    print("Warning: financials_features.parquet failed to load. Skipping.", str(e))
    fin_features = pd.DataFrame(columns=['UniqueID'])

print("Loading and aggregating transactions (this takes a moment)...")
txn = pd.read_parquet('transactions_features')

# Enhanced feature engineering
# 1. Base aggregations
txn_base = txn.groupby('UniqueID').agg(
    total_txn_count=('TransactionAmount', 'count'),
    mean_txn_amount=('TransactionAmount', 'mean'),
    sum_txn_amount=('TransactionAmount', 'sum'),
    std_txn_amount=('TransactionAmount', 'std'),
).reset_index()

# 2. Seasonality (Nov, Dec, Jan) transactions
# The training window includes historical data, so we check the month of TransactionDate
txn['Month'] = pd.to_datetime(txn['TransactionDate']).dt.month
holiday_mask = txn['Month'].isin([11, 12, 1])
holiday_counts = txn[holiday_mask].groupby('UniqueID').size().reset_index(name='holiday_txn_count')
non_holiday_counts = txn[~holiday_mask].groupby('UniqueID').size().reset_index(name='non_holiday_txn_count')

# 3. Debit / Credit split
debits = txn[txn['IsDebitCredit'] == 'D'].groupby('UniqueID').agg(
    debit_count=('TransactionAmount', 'count'),
    debit_sum=('TransactionAmount', 'sum')
).reset_index()

credits = txn[txn['IsDebitCredit'] == 'C'].groupby('UniqueID').agg(
    credit_count=('TransactionAmount', 'count'),
    credit_sum=('TransactionAmount', 'sum')
).reset_index()

# Merge all txn features
txn_features = txn_base.merge(holiday_counts, on='UniqueID', how='left')\
                       .merge(non_holiday_counts, on='UniqueID', how='left')\
                       .merge(debits, on='UniqueID', how='left')\
                       .merge(credits, on='UniqueID', how='left')

txn_features.fillna(0, inplace=True)

print("Merging all features...")
train_df = train.merge(demo, on='UniqueID', how='left')\
                .merge(fin_features, on='UniqueID', how='left')\
                .merge(txn_features, on='UniqueID', how='left')
                
test_df = test.merge(demo, on='UniqueID', how='left')\
              .merge(fin_features, on='UniqueID', how='left')\
              .merge(txn_features, on='UniqueID', how='left')

# Drop unused
drop_cols = ['UniqueID', 'BirthDate', 'ResidentialCityName']
X = train_df.drop(columns=drop_cols + ['next_3m_txn_count'], errors='ignore')
y = train_df['next_3m_txn_count']
X_test = test_df.drop(columns=drop_cols, errors='ignore')

# Categoricals to dummies
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f"Training advanced model with {X.shape[1]} features...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
model.fit(X_train, y_train)

preds = np.clip(model.predict(X_val), 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
print(f"Validation RMSLE: {rmsle:.4f}")

# Final train
model.fit(X, y)
test_preds = np.clip(model.predict(X_test), 0, None)

sub = pd.DataFrame({'UniqueID': test['UniqueID'], 'next_3m_txn_count': np.round(test_preds).astype(int)})
sub.to_csv('advanced_submission.csv', index=False)

joblib.dump({'model': model, 'features': list(X.columns), 'drop_cols': drop_cols}, 'model_pipeline.pkl')
print("Model trained and saved to model_pipeline.pkl")
