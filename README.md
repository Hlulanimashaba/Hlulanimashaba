# 🏦 Nedbank VolumeAI — Transaction Forecasting Dashboard

**Developed By: Mashaba Hlulani Charles**

An AI-powered web application that predicts the total number of bank transactions each customer will make in the next 3-month window (November 2015 – January 2016), built for the Nedbank Transaction Volumetrics Challenge.

---

##  Challenge Overview

| Detail | Value |
|---|---|
| **Objective** | Predict `next_3m_txn_count` for each customer |
| **Metric** | RMSLE (Root Mean Squared Logarithmic Error) |
| **Model** | RandomForestRegressor (150 trees, max_depth=16) |
| **Validation RMSLE** | ~0.5944 |
| **Training Customers** | 8,360 |
| **Test Customers** | 3,584 |

---

##  How to Run the App

### Prerequisites

The **only** requirement is **Python 3.8+** installed on the machine.

- Download Python from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
-  **Windows users**: During installation, check **"Add Python to PATH"**

**No other manual installation is needed.** All dependencies install automatically on first run.

---

### Option 1: One-Click Launcher (Recommended)

#### Windows:
1. Copy the entire `Nedbank_Challenge` folder to the target machine
2. **Double-click `run_app.bat`**
3. Wait for automatic setup (first run only)
4. Open your browser to **http://localhost:5000**

#### Mac / Linux:
1. Copy the entire `Nedbank_Challenge` folder to the target machine
2. Open a terminal and navigate to the folder
3. Run:
   ```bash
   bash run_app.sh
   ```
4. Open your browser to **http://localhost:5000**

---

### Option 2: Run Directly with Python

Open a terminal/command prompt in the project folder and run:

```bash
# Windows (if Python launcher is installed)
py app.py

# Windows / Mac / Linux (if python is in PATH)
python app.py

# Mac / Linux alternative
python3 app.py
```

The app will **automatically install** any missing packages (`flask`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyarrow`) on first run.

Then open your browser to: **http://localhost:5000**

---

### Option 3: Manual Setup (Advanced)

If you prefer to set up a virtual environment manually:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

##  Using the Dashboard

Once the server is running at **http://localhost:5000**:

1. **Dashboard Tab** — Select a customer from the dropdown and click **"Forecast Future"** to see:
   - Predicted transaction count for the next 3 months
   - Model confidence level and error margin
   - Customer demographics and profile
   - Advanced behavioral indicators (holiday transactions, credits/debits)
   - Historical transaction volume chart

2. **Models & Confidence Tab** — View:
   - Model architecture details and hyperparameters
   - Validation RMSLE score
   - Top 10 feature importances chart

---

##  Project Files

### Application Files
| File | Description |
|---|---|
| `app.py` | Flask web application (main entry point) |
| `run_app.bat` | Windows one-click launcher (auto-installs everything) |
| `run_app.sh` | Mac/Linux one-click launcher (auto-installs everything) |
| `requirements.txt` | Python package dependencies |
| `templates/index.html` | Dashboard HTML template |
| `static/styles.css` | Dashboard styling |
| `static/app.js` | Dashboard JavaScript logic |

### Model & Training Files
| File | Description |
|---|---|
| `model_pipeline.pkl` | Pre-trained model pipeline (ready to use) |
| `train_model.py` | Baseline model training script |
| `advanced_train_model.py` | Advanced model with extra features |
| `master_train_model.py` | Full-featured master training pipeline |
| `evaluate.py` | Local RMSLE scoring script |

### Data Files
| File | Description |
|---|---|
| `Train.csv` | Training labels (8,360 customers) |
| `Test.csv` | Test customer IDs (3,584 customers) |
| `SampleSubmission.csv` | Submission template |
| `transactions_features/` | 18M transaction rows (Dec 2012 – Oct 2015, Parquet) |
| `financials_features.parquet` | 372K financial snapshot rows (Dec 2013 – Oct 2015) |
| `demographics_clean/` | 11,944 customer profiles (Parquet) |
| `VariableDefinitions.csv` | Data dictionary for all columns |

### Submission Files
| File | Description |
|---|---|
| `my_baseline_submission.csv` | Baseline model predictions |
| `advanced_submission.csv` | Advanced model predictions |
| `master_submission.csv` | Master pipeline predictions (best) |

---

## 🔧 Dependencies (Auto-Installed)

| Package | Purpose |
|---|---|
| `flask` | Web server framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Machine learning (RandomForest model) |
| `joblib` | Model serialization/loading |
| `pyarrow` | Reading Parquet data files |

---

## 🧪 Re-Training the Model

If you want to retrain the model from scratch:

```bash
# Baseline model
python train_model.py

# Advanced model (more features)
python advanced_train_model.py

# Master pipeline (full features, best performance)
python master_train_model.py
```

This will regenerate `model_pipeline.pkl` and the corresponding submission CSV.

---

## 📊 Local Scoring

To evaluate predictions locally (requires a reference file):

```bash
python evaluate.py my_submission.csv PublicReference.csv
```

---

## 📝 Data Notes

- All transaction dates are **before** the prediction window (max: Oct 2015)
- 567 of 11,944 customers have **no financials data** — handled with left joins
- `BirthDate` contains data quality challenges — inspect before using
- `AnnualGrossIncome` is null for ~6% of customers
- `TransactionAmount` is signed: negative = debit, positive = credit
- The prediction window (Nov–Jan) includes **holiday seasonality**

---

## 🔗 Important Links

- **Zindi Challenge Page**: Submit predictions on the Zindi platform
- **Nedbank Registration**: [http://register.data.challenge.nedbank.co.za/](http://register.data.challenge.nedbank.co.za/)

---

## 📄 License

CC-BY SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)
