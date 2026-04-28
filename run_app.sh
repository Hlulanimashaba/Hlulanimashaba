#!/bin/bash
echo ""
echo "============================================"
echo " Nedbank VolumeAI - Transaction Forecasting"
echo " Developed By Mashaba Hlulani Charles"
echo "============================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python is not installed."
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    exit 1
fi

echo "[INFO] Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[Setup] Creating virtual environment (first-time setup)..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "[WARNING] Failed to create venv. Running directly..."
        echo ""
        echo "[INFO] Starting Nedbank VolumeAI server..."
        echo "[INFO] Open your browser to: http://localhost:5000"
        echo "[INFO] Press Ctrl+C to stop the server."
        echo ""
        $PYTHON_CMD app.py
        exit 0
    fi
    echo "[Setup] Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "[Setup] Checking dependencies..."
pip install --quiet -r requirements.txt 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Retrying installs..."
    pip install --quiet flask pandas numpy scikit-learn joblib pyarrow
fi

echo ""
echo "[INFO] Starting Nedbank VolumeAI server..."
echo "[INFO] Open your browser to: http://localhost:5000"
echo "[INFO] Press Ctrl+C to stop the server."
echo ""
python app.py
