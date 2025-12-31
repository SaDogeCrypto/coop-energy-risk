#!/bin/bash
# Quick start script for Coop Energy Risk Demo

echo "======================================"
echo "Coop Energy Risk Demo - Quick Start"
echo "======================================"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Generate sample data
echo ""
echo "Generating synthetic data..."
python -c "
import sys
sys.path.insert(0, 'src')
from data_fetcher import generate_synthetic_lmp, save_data
from load_profiles import generate_coop_load_profile, save_load_profile

lmp_df = generate_synthetic_lmp(days=365, nodes=['COOP_NODE_1', 'COOP_NODE_2', 'COOP_NODE_3', 'HUB_SPP'])
save_data(lmp_df, 'synthetic_lmp_hourly.parquet')

load_df = generate_coop_load_profile(days=365, base_load_mw=50, peak_load_mw=100)
save_load_profile(load_df, 'medium_mixed')
print('Data generated successfully!')
"

# Run dashboard
echo ""
echo "Starting dashboard..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app/dashboard.py
