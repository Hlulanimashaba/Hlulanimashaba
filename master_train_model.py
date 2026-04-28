import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

print("--- NEDBANK VOLUMETRICS ML PIPELINE ---")
print("1. Loading raw inputs...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
demo = pd.read_parquet('demographics_clean')
sample_sub = pd.read_csv('SampleSubmission.csv')

print("2. Processing 372K Financial Snapshots...")
try:
    fin = pd.read_parquet('financials_features.parquet')
    fin_features = fin.groupby('UniqueID').agg(
        fin_mean_NII=('NetInterestIncome', 'mean'),
        fin_sum_NII=('NetInterestIncome', 'sum'),
        fin_mean_NIR=('NetInterestRevenue', 'mean'),
        fin_sum_NIR=('NetInterestRevenue', 'sum')
    ).reset_index()
    
    # Financial product counts
    fin_prods = fin.groupby(['UniqueID', 'Product']).size().unstack(fill_value=0).reset_index()
    fin_prods.columns = ['UniqueID'] + [f"fin_prod_{c.replace(' ', '_')}" for c in fin_prods.columns[1:]]
    fin_features = fin_features.merge(fin_prods, on='UniqueID', how='left')
except Exception as e:
    print(f"Warning: Failed financials load ({e})")
    fin_features = pd.DataFrame(columns=['UniqueID'])

print("3. Processing 18M Transactions... (Heavy computation)")
txn = pd.read_parquet('transactions_features')

# Numeric basic aggregates
txn_base = txn.groupby('UniqueID').agg(
    total_txn_count=('TransactionAmount', 'count'),
    mean_txn_amount=('TransactionAmount', 'mean'),
    sum_txn_amount=('TransactionAmount', 'sum'),
    std_txn_amount=('TransactionAmount', 'std'),
    max_balance=('StatementBalance', 'max'),
    min_balance=('StatementBalance', 'min'),
    mean_balance=('StatementBalance', 'mean')
).reset_index()

# Seasonality
txn['Date'] = pd.to_datetime(txn['TransactionDate'])
txn['Month'] = txn['Date'].dt.month
holiday_mask = txn['Month'].isin([11, 12, 1])
holiday_counts = txn[holiday_mask].groupby('UniqueID').size().reset_index(name='holiday_txn_count')
non_holiday_counts = txn[~holiday_mask].groupby('UniqueID').size().reset_index(name='non_holiday_txn_count')

# Directional Flows
debits = txn[txn['IsDebitCredit'] == 'D'].groupby('UniqueID').agg(debit_count=('TransactionAmount', 'count'), debit_sum=('TransactionAmount', 'sum')).reset_index()
credits = txn[txn['IsDebitCredit'] == 'C'].groupby('UniqueID').agg(credit_count=('TransactionAmount', 'count'), credit_sum=('TransactionAmount', 'sum')).reset_index()

# Categorical Unstacking directly translating to behavioral patterns
print("   -> Extracting categorical behavior patterns...")
txn['TransactionTypeDescription'] = txn['TransactionTypeDescription'].fillna('Unknown')
txn_types = txn.groupby(['UniqueID', 'TransactionTypeDescription']).size().unstack(fill_value=0).reset_index()
txn_types.columns = ['UniqueID'] + [f"txn_type_{str(c).replace(' ', '_')}" for c in txn_types.columns[1:]]

txn['TransactionBatchDescription'] = txn['TransactionBatchDescription'].fillna('Unknown')
txn_batch = txn.groupby(['UniqueID', 'TransactionBatchDescription']).size().unstack(fill_value=0).reset_index()
txn_batch.columns = ['UniqueID'] + [f"txn_batch_{str(c).replace(' ', '_')}" for c in txn_batch.columns[1:]]

txn['ReversalTypeDescription'] = txn['ReversalTypeDescription'].fillna('None')
txn_rev = txn.groupby(['UniqueID', 'ReversalTypeDescription']).size().unstack(fill_value=0).reset_index()
txn_rev.columns = ['UniqueID'] + [f"txn_rev_{str(c).replace(' ', '_')}" for c in txn_rev.columns[1:]]

# Combine all transaction features
print("4. Consolidating unified behavioral matrix...")
txn_features = txn_base.merge(holiday_counts, on='UniqueID', how='left')\
                       .merge(non_holiday_counts, on='UniqueID', how='left')\
                       .merge(debits, on='UniqueID', how='left')\
                       .merge(credits, on='UniqueID', how='left')\
                       .merge(txn_types, on='UniqueID', how='left')\
                       .merge(txn_batch, on='UniqueID', how='left')\
                       .merge(txn_rev, on='UniqueID', how='left')
txn_features.fillna(0, inplace=True)

print("5. Merging Global Datasets...")
train_df = train.merge(demo, on='UniqueID', how='left')\
                .merge(fin_features, on='UniqueID', how='left')\
                .merge(txn_features, on='UniqueID', how='left')
                
test_df = test.merge(demo, on='UniqueID', how='left')\
              .merge(fin_features, on='UniqueID', how='left')\
              .merge(txn_features, on='UniqueID', how='left')

# Exclude non-features
drop_cols = ['UniqueID', 'BirthDate', 'ResidentialCityName', 'CountryCodeNationality', 'AccountID_x', 'AccountID_y', 'AccountID', 'Date']
X = train_df.drop(columns=drop_cols + ['next_3m_txn_count'], errors='ignore')
y = train_df['next_3m_txn_count']
X_test = test_df.drop(columns=drop_cols, errors='ignore')

# Automatic Dummy Variable Encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f"6. Training Deep Random Forest ({X.shape[1]} dimensional features)...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, max_depth=16, min_samples_split=5)
model.fit(X_train, y_train)

# Strict bounding > 0
preds = np.clip(model.predict(X_val), 0, None)
rmsle = np.sqrt(mean_squared_log_error(y_val, preds))
print(f"-> Cross-Validation RMSLE Achieved: {rmsle:.4f}")

# Train absolute finalized model explicitly mapped on 100% of labeled data
print("7. Refitting on 100% of data for optimal output...")
model.fit(X, y)
test_preds = np.clip(model.predict(X_test), 0, None)

sub = pd.DataFrame({'UniqueID': test['UniqueID'], 'next_3m_txn_count': np.round(test_preds).astype(int)})
# Safely guarantee the structure relies ONLY explicitly on SampleSubmission.csv structure:
sub = sample_sub[['UniqueID']].merge(sub, on='UniqueID', how='left')
sub['next_3m_txn_count'] = sub['next_3m_txn_count'].fillna(0).astype(int)

sub.to_csv('master_submission.csv', index=False)

joblib.dump({
    'model': model, 
    'features': list(X.columns), 
    'drop_cols': drop_cols
}, 'model_pipeline.pkl')

print("SUCCESS: Pipeline exported. File 'master_submission.csv' is ready for Zindi submission!")
