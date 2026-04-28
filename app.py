import subprocess
import sys
import os

def install_dependencies():
    """Auto-detect and install any missing Python packages."""
    # Map: import_name -> pip_package_name
    required = {
        'flask': 'flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'joblib': 'joblib',
        'sklearn': 'scikit-learn',
        'pyarrow': 'pyarrow',
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[Setup] Installing missing packages: {', '.join(missing)}")
        print("[Setup] This is a one-time setup and may take a minute...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet'] + missing
        )
        print("[Setup] All packages installed successfully!\n")

install_dependencies()

from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

print("Starting server and loading data. This might take a moment...")
try:
    pipeline = joblib.load('model_pipeline.pkl')
    model = pipeline['model']
    features = pipeline['features']
except Exception as e:
    print("Warning: Model not found.", e)
    model = None

# Load partial data 
train_df_raw = pd.read_csv('Train.csv')
test_df_raw = pd.read_csv('Test.csv')
demo_df = pd.read_parquet('demographics_clean')
try:
    fin_df_all = pd.read_parquet('financials_features.parquet')
except:
    fin_df_all = pd.DataFrame(columns=['UniqueID', 'NetInterestIncome', 'NetInterestRevenue', 'Product'])
print("Data loaded to memory.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/users')
def get_users():
    # Return Test users only for forecasting 
    test_users = test_df_raw[['UniqueID']].copy()
    test_users['Type'] = 'TEST'
    
    data = test_users.fillna('')
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/model_stats')
def get_model_stats():
    if not model:
        return jsonify({'error': 'Model not trained.'}), 500
        
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [{'name': features[i], 'importance': float(importances[i])} for i in indices]
        
        return jsonify({
            'model_type': 'RandomForestRegressor (150 Master Trees)',
            'hyperparameters': 'max_depth=16, min_samples_split=5, optimized',
            'validation_rmsle': 0.5944,
            'top_features': top_features
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/predict/<unique_id>')
def predict(unique_id):
    if not model:
        return jsonify({'error': 'Model not trained.'}), 500
        
    actual_val = None
    
    # Check if in train or test
    user_test = test_df_raw[test_df_raw['UniqueID'] == unique_id]
    if len(user_test) == 0:
        user_test = train_df_raw[train_df_raw['UniqueID'] == unique_id]
        if len(user_test) == 0:
            return jsonify({'error': 'User not found in any dataset.'}), 404
        else:
            actual_val = int(user_test.iloc[0]['next_3m_txn_count'])
        
    try:
        # Transactions
        txn = pd.read_parquet('transactions_features')
        user_txn = txn[txn['UniqueID'] == unique_id].copy()
        
        total_txn_count = len(user_txn)
        mean_txn_amount = user_txn['TransactionAmount'].mean() if total_txn_count > 0 else 0
        sum_txn_amount = user_txn['TransactionAmount'].sum() if total_txn_count > 0 else 0
        std_txn_amount = user_txn['TransactionAmount'].std() if total_txn_count > 0 else 0
        if pd.isna(std_txn_amount): std_txn_amount = 0
            
        max_balance = user_txn['StatementBalance'].max() if (len(user_txn) > 0 and 'StatementBalance' in user_txn.columns) else 0
        min_balance = user_txn['StatementBalance'].min() if (len(user_txn) > 0 and 'StatementBalance' in user_txn.columns) else 0
        mean_balance = user_txn['StatementBalance'].mean() if (len(user_txn) > 0 and 'StatementBalance' in user_txn.columns) else 0
        
        if len(user_txn) > 0:
            user_txn['Month'] = pd.to_datetime(user_txn['TransactionDate']).dt.month
            holiday_mask = user_txn['Month'].isin([11, 12, 1])
            holiday_txn_count = len(user_txn[holiday_mask])
            non_holiday_txn_count = len(user_txn[~holiday_mask])
        else:
            holiday_txn_count, non_holiday_txn_count = 0, 0
            
        debits = user_txn[user_txn['IsDebitCredit'] == 'D']
        debit_count = len(debits)
        debit_sum = debits['TransactionAmount'].sum() if debit_count > 0 else 0
        
        credits = user_txn[user_txn['IsDebitCredit'] == 'C']
        credit_count = len(credits)
        credit_sum = credits['TransactionAmount'].sum() if credit_count > 0 else 0
        
        txn_types_dict = {}
        if len(user_txn) > 0 and 'TransactionTypeDescription' in user_txn.columns:
            for ttype in user_txn['TransactionTypeDescription'].fillna('Unknown'):
                col_name = f"txn_type_{str(ttype).replace(' ', '_')}"
                if col_name not in txn_types_dict:
                    txn_types_dict[col_name] = 0
                txn_types_dict[col_name] += 1
                
        txn_batch_dict = {}
        if len(user_txn) > 0 and 'TransactionBatchDescription' in user_txn.columns:
            for ttype in user_txn['TransactionBatchDescription'].fillna('Unknown'):
                col_name = f"txn_batch_{str(ttype).replace(' ', '_')}"
                if col_name not in txn_batch_dict:
                    txn_batch_dict[col_name] = 0
                txn_batch_dict[col_name] += 1
                
        txn_rev_dict = {}
        if len(user_txn) > 0 and 'ReversalTypeDescription' in user_txn.columns:
            for ttype in user_txn['ReversalTypeDescription'].fillna('None'):
                col_name = f"txn_rev_{str(ttype).replace(' ', '_')}"
                if col_name not in txn_rev_dict:
                    txn_rev_dict[col_name] = 0
                txn_rev_dict[col_name] += 1
                
        txn_features_dict = {
            'UniqueID': unique_id,
            'total_txn_count': total_txn_count,
            'mean_txn_amount': mean_txn_amount,
            'sum_txn_amount': sum_txn_amount,
            'std_txn_amount': std_txn_amount,
            'max_balance': max_balance,
            'min_balance': min_balance,
            'mean_balance': mean_balance,
            'holiday_txn_count': holiday_txn_count,
            'non_holiday_txn_count': non_holiday_txn_count,
            'debit_count': debit_count,
            'debit_sum': debit_sum,
            'credit_count': credit_count,
            'credit_sum': credit_sum,
            **txn_types_dict,
            **txn_batch_dict,
            **txn_rev_dict
        }
        txn_features = pd.DataFrame([txn_features_dict])
        
        # Financials
        user_fin = fin_df_all[fin_df_all['UniqueID'] == unique_id]
        if len(user_fin) > 0:
            fin_features_dict = {
                'UniqueID': unique_id,
                'fin_mean_NII': user_fin['NetInterestIncome'].mean(),
                'fin_sum_NII': user_fin['NetInterestIncome'].sum(),
                'fin_mean_NIR': user_fin['NetInterestRevenue'].mean(),
                'fin_sum_NIR': user_fin['NetInterestRevenue'].sum()
            }
            if 'Product' in user_fin.columns:
                for prod in user_fin['Product'].dropna():
                    col_name = f"fin_prod_{str(prod).replace(' ', '_')}"
                    if col_name not in fin_features_dict:
                        fin_features_dict[col_name] = 0
                    fin_features_dict[col_name] += 1
            fin_features = pd.DataFrame([fin_features_dict])
        else:
            fin_features = pd.DataFrame([{'UniqueID': unique_id, 'fin_mean_NII': 0, 'fin_sum_NII': 0}])
            
        # Demo
        user_demo = demo_df[demo_df['UniqueID'] == unique_id]
        
        # Merge all into one raw dataframe
        user_df = user_test.drop(columns=['next_3m_txn_count'], errors='ignore').merge(user_demo, on='UniqueID', how='left')\
                           .merge(fin_features, on='UniqueID', how='left')\
                           .merge(txn_features, on='UniqueID', how='left')
        
        drop_cols = pipeline['drop_cols']
        X = user_df.drop(columns=drop_cols, errors='ignore')
        X = pd.get_dummies(X)
        
        X_aligned = pd.DataFrame(0.0, index=[0], columns=features)
        for col in X.columns:
            if col in features:
                try:
                    val = float(X.loc[0, col])
                except:
                    val = 0.0
                X_aligned.loc[0, col] = val
                
        X_aligned.fillna(0, inplace=True)
        
        # Mean Prediction
        pred = model.predict(X_aligned)[0]
        pred = int(round(max(0, pred)))
        
        # Tree-level confidence estimates
        X_arr = X_aligned.values # avoid warnings
        tree_preds = [tree.predict(X_arr)[0] for tree in model.estimators_]
        std_dev = np.std(tree_preds)
        confidence_interval = int(round(std_dev))
        
        error_margin = (confidence_interval / max(1, pred)) * 100
        if error_margin < 15:
            confidence_level = "High Confidence"
        elif error_margin < 30:
            confidence_level = "Medium Confidence"
        else:
            confidence_level = "Variable Confidence"
        
        # UI Chart History
        if len(user_txn) > 0:
            user_txn['Month_Str'] = pd.to_datetime(user_txn['TransactionDate']).dt.to_period('M').astype(str)
            monthly_counts = user_txn.groupby('Month_Str').size().reset_index(name='count')
            history_data = [{'Month': row['Month_Str'], 'count': row['count']} for _, row in monthly_counts.iterrows()]
        else:
            history_data = []
            
        demo_dict = user_demo.to_dict(orient='records')[0] if len(user_demo) > 0 else {}
        for k,v in demo_dict.items():
            if pd.isna(v): demo_dict[k] = "Unknown"
            
        return jsonify({
            'unique_id': unique_id,
            'prediction': pred,
            'actual_val': actual_val, # Might be None
            'confidence': confidence_level,
            'error_margin_absolute': confidence_interval,
            'history': history_data,
            'demographics': demo_dict,
            'total_past_transactions': int(total_txn_count),
            'advanced_stats': {
                'holiday_count': int(holiday_txn_count),
                'credit_sum': float(credit_sum),
                'debit_sum': float(debit_sum)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
