# Coop Energy Risk Demo

A prototype dashboard demonstrating energy price risk analysis for electric cooperatives.

## Project Structure

```
coop-energy-risk/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Raw LMP data from Grid Status
│   └── processed/        # Cleaned data ready for analysis
├── src/
│   ├── data_fetcher.py   # Grid Status data pipeline
│   ├── load_profiles.py  # Synthetic coop load generation
│   ├── risk_models.py    # Monte Carlo, VaR calculations
│   └── hedge_analysis.py # Hedge scenario comparisons
├── app/
│   └── dashboard.py      # Streamlit dashboard
└── notebooks/
    └── exploration.ipynb # Data exploration
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch sample data (SPP, last 90 days)
python src/data_fetcher.py

# Run the dashboard
streamlit run app/dashboard.py
```

## 3-Day Build Plan

### Day 1: Data
- [ ] Get gridstatus library working
- [ ] Pull 1 year historical LMPs for SPP or ERCOT
- [ ] Identify 3-5 relevant nodes
- [ ] Basic data exploration

### Day 2: Models
- [ ] Create synthetic coop load profile
- [ ] Historical cost calculation (load × price)
- [ ] Monte Carlo price simulation
- [ ] Simple hedge comparison (fixed swap vs spot)

### Day 3: Dashboard
- [ ] LMP explorer page
- [ ] Load profile visualization
- [ ] Risk analysis output
- [ ] Hedge scenario comparison

## Key Concepts

### LMP (Locational Marginal Price)
The price of electricity at a specific node, updated every 5-15 minutes.

### Load Profile
A coop's hourly electricity demand pattern over time.

### Risk Metrics
- **Cost-at-Risk (CaR)**: What's the worst-case cost at 95% confidence?
- **Expected Cost**: Average cost across simulations
- **Volatility**: Standard deviation of costs

### Hedge Scenarios
- **No hedge**: Buy at spot prices
- **Fixed swap**: Lock in a fixed price for all MWh
- **Collar**: Cap upside exposure, give up some downside benefit
