import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib

print("Loading data...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
demo = pd.read_parquet('demographics_clean')

print("Loading transactions... This might take a bit.")
# Reading 18M rows can be memory intensive. We only need simple aggregations for baseline.
txn = pd.read_parquet('transactions_features')

# Grouping transactions to get features
print("Aggregating transactions...")
txn_features = txn.groupby('UniqueID').agg(
    total_txn_count=('TransactionAmount', 'count'),
    mean_txn_amount=('TransactionAmount', 'mean'),
    sum_txn_amount=('TransactionAmount', 'sum'),
    std_txn_amount=('TransactionAmount', 'std'),
).reset_index()
txn_features['std_txn_amount'].fillna(0, inplace=True)

print("Merging features...")
train_df = train.merge(demo, on='UniqueID', how='left').merge(txn_features, on='UniqueID', how='left')
test_df = test.merge(demo, on='UniqueID', how='left').merge(txn_features, on='UniqueID', how='left')

# Drop columns that are tricky for baseline (Dates, high cardinality strings)
drop_cols = ['UniqueID', 'BirthDate', 'ResidentialCityName']
X = train_df.drop(columns=drop_cols + ['next_3m_txn_count'])
y = train_df['next_3m_txn_count']
X_test = test_df.drop(columns=drop_cols)

# One-hot encode categoricals simply
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# Align columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Fill missing values
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print("Training model...")
# To predict count we can use a regressor. Target is count >= 0.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Validate
preds = model.predict(X_val)
# Ensure no negative predictions
preds = np.clip(preds, 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
print(f"Validation RMSLE: {rmsle:.4f}")

# Train on all data for final model
model.fit(X, y)
test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)

# Save submission
sub = pd.DataFrame({
    'UniqueID': test['UniqueID'],
    'next_3m_txn_count': np.round(test_preds).astype(int)
})
sub.to_csv('my_baseline_submission.csv', index=False)

# Save model and feature columns
joblib.dump({
    'model': model,
    'features': list(X.columns),
    'drop_cols': drop_cols
}, 'model_pipeline.pkl')
print("Model trained and saved to model_pipeline.pkl")
print("Submission saved to my_baseline_submission.csv")
