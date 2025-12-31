"""
Data fetcher for LMP data using Grid Status library.

Grid Status provides open source scrapers for major ISOs:
- SPP (Southwest Power Pool) - heavy coop territory
- ERCOT (Texas)
- MISO, PJM, CAISO, NYISO, ISONE

For the demo, we focus on SPP.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# gridstatus is optional - only imported when fetching real data
# import gridstatus

# Create data directories
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for dir in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    dir.mkdir(parents=True, exist_ok=True)


def fetch_spp_lmp(days_back: int = 365) -> pd.DataFrame:
    """
    Fetch historical LMP data from SPP.
    
    SPP covers: Kansas, Oklahoma, Nebraska, parts of surrounding states
    Heavy cooperative territory.
    """
    import gridstatus
    
    print("Initializing SPP connection...")
    spp = gridstatus.SPP()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Fetching LMP data from {start_date.date()} to {end_date.date()}...")
    
    # Get real-time LMP data
    # This returns settlement point prices
    lmp_df = spp.get_lmp(
        start=start_date,
        end=end_date,
        market="REAL_TIME_5MIN",  # or "DAY_AHEAD_HOURLY"
    )
    
    print(f"Fetched {len(lmp_df):,} rows")
    return lmp_df


def fetch_ercot_lmp(days_back: int = 365) -> pd.DataFrame:
    """
    Fetch historical LMP data from ERCOT.
    
    ERCOT covers: Most of Texas
    """
    import gridstatus
    
    print("Initializing ERCOT connection...")
    ercot = gridstatus.Ercot()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Fetching LMP data from {start_date.date()} to {end_date.date()}...")
    
    lmp_df = ercot.get_lmp(
        start=start_date,
        end=end_date,
        market="REAL_TIME_15MIN",
    )
    
    print(f"Fetched {len(lmp_df):,} rows")
    return lmp_df


def get_available_nodes(lmp_df: pd.DataFrame) -> list:
    """Get list of available pricing nodes from LMP data."""
    if "Location" in lmp_df.columns:
        return sorted(lmp_df["Location"].unique())
    elif "Settlement Point" in lmp_df.columns:
        return sorted(lmp_df["Settlement Point"].unique())
    else:
        print(f"Columns available: {lmp_df.columns.tolist()}")
        return []


def filter_to_nodes(lmp_df: pd.DataFrame, nodes: list) -> pd.DataFrame:
    """Filter LMP data to specific nodes."""
    if "Location" in lmp_df.columns:
        return lmp_df[lmp_df["Location"].isin(nodes)]
    elif "Settlement Point" in lmp_df.columns:
        return lmp_df[lmp_df["Settlement Point"].isin(nodes)]
    return lmp_df


def resample_to_hourly(lmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample sub-hourly data to hourly averages.
    Coops typically deal with hourly data.
    """
    # Identify the time and price columns
    time_col = None
    for col in ["Interval Start", "Time", "Datetime"]:
        if col in lmp_df.columns:
            time_col = col
            break
    
    if time_col is None:
        print(f"Could not find time column. Available: {lmp_df.columns.tolist()}")
        return lmp_df
    
    lmp_df = lmp_df.copy()
    lmp_df[time_col] = pd.to_datetime(lmp_df[time_col])
    lmp_df["hour"] = lmp_df[time_col].dt.floor("H")
    
    # Group by hour and location, average the LMP
    location_col = "Location" if "Location" in lmp_df.columns else "Settlement Point"
    price_col = "LMP" if "LMP" in lmp_df.columns else lmp_df.select_dtypes(include='number').columns[0]
    
    hourly = lmp_df.groupby(["hour", location_col])[price_col].mean().reset_index()
    hourly.columns = ["datetime", "node", "lmp"]
    
    return hourly


def save_data(df: pd.DataFrame, filename: str):
    """Save dataframe to parquet."""
    filepath = PROCESSED_DIR / filename
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath}")


def load_data(filename: str) -> pd.DataFrame:
    """Load dataframe from parquet."""
    filepath = PROCESSED_DIR / filename
    return pd.read_parquet(filepath)


# ============================================================
# DEMO DATA GENERATION (if Grid Status API is slow/unavailable)
# ============================================================

def generate_synthetic_lmp(
    days: int = 365,
    nodes: list = ["NODE_A", "NODE_B", "NODE_C"],
    base_price: float = 35.0,  # $/MWh
) -> pd.DataFrame:
    """
    Generate synthetic LMP data for demo purposes.
    
    Incorporates realistic patterns:
    - Seasonal variation (higher in summer/winter)
    - Daily patterns (peaks in afternoon)
    - Random volatility
    - Occasional price spikes
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Generate hourly timestamps
    start = datetime.now() - timedelta(days=days)
    hours = pd.date_range(start=start, periods=days * 24, freq="H")
    
    records = []
    
    for node in nodes:
        # Node-specific basis (some nodes more expensive)
        node_basis = np.random.uniform(-5, 5)
        
        for i, hour in enumerate(hours):
            # Seasonal component (higher in summer/winter)
            day_of_year = hour.timetuple().tm_yday
            seasonal = 10 * np.sin(2 * np.pi * (day_of_year - 30) / 365)  # Peak in summer
            seasonal += 5 * np.cos(4 * np.pi * day_of_year / 365)  # Secondary winter peak
            
            # Daily component (peaks 2-6 PM)
            hour_of_day = hour.hour
            daily = 8 * np.sin(np.pi * (hour_of_day - 6) / 12) if 6 <= hour_of_day <= 22 else -5
            
            # Random volatility
            volatility = np.random.normal(0, 5)
            
            # Occasional price spikes (1% chance)
            spike = 0
            if np.random.random() < 0.01:
                spike = np.random.exponential(50)
            
            # Combine components
            lmp = base_price + node_basis + seasonal + daily + volatility + spike
            lmp = max(lmp, -10)  # Floor (negative prices happen but have limits)
            
            records.append({
                "datetime": hour,
                "node": node,
                "lmp": round(lmp, 2)
            })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Try to fetch real data, fall back to synthetic
    try:
        print("Attempting to fetch real SPP data...")
        lmp_df = fetch_spp_lmp(days_back=90)  # Start with 90 days
        
        # Explore available nodes
        nodes = get_available_nodes(lmp_df)
        print(f"\nFound {len(nodes)} nodes. Sample: {nodes[:10]}")
        
        # Resample to hourly
        hourly_df = resample_to_hourly(lmp_df)
        save_data(hourly_df, "spp_lmp_hourly.parquet")
        
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("\nGenerating synthetic data for demo...")
        
        synthetic_df = generate_synthetic_lmp(
            days=365,
            nodes=["COOP_NODE_1", "COOP_NODE_2", "COOP_NODE_3", "HUB_SPP"]
        )
        save_data(synthetic_df, "synthetic_lmp_hourly.parquet")
        print(f"\nGenerated {len(synthetic_df):,} rows of synthetic data")
        
        # Preview
        print("\nSample data:")
        print(synthetic_df.head(10))
        print(f"\nPrice stats by node:")
        print(synthetic_df.groupby("node")["lmp"].describe())
