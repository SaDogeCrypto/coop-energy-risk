"""
Load Profile Generator

Creates synthetic load profiles that mimic typical rural electric cooperative patterns.

Characteristics of rural coop load:
- Residential heavy (farms, rural homes)
- Agricultural load (irrigation, grain drying - seasonal)
- Summer peaks (A/C)
- Winter peaks (heating in some regions)
- Lower industrial base load than urban utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def generate_coop_load_profile(
    days: int = 365,
    base_load_mw: float = 50.0,  # Average load in MW
    peak_load_mw: float = 100.0,  # Summer peak
    coop_type: str = "mixed",  # "residential", "agricultural", "mixed"
) -> pd.DataFrame:
    """
    Generate a synthetic hourly load profile for a rural electric cooperative.
    
    Args:
        days: Number of days to generate
        base_load_mw: Base/minimum load in MW
        peak_load_mw: Expected summer peak in MW
        coop_type: Type of coop affects load shape
        
    Returns:
        DataFrame with datetime and load_mw columns
    """
    np.random.seed(42)
    
    start = datetime.now() - timedelta(days=days)
    hours = pd.date_range(start=start, periods=days * 24, freq="H")
    
    loads = []
    
    for hour in hours:
        day_of_year = hour.timetuple().tm_yday
        hour_of_day = hour.hour
        day_of_week = hour.weekday()  # 0 = Monday
        
        # Base load
        load = base_load_mw
        
        # Seasonal component
        # Summer peak (day ~180 = July), secondary winter peak
        summer_factor = np.sin(2 * np.pi * (day_of_year - 90) / 365)  # Peak in summer
        winter_factor = 0.5 * np.cos(2 * np.pi * day_of_year / 365)  # Secondary winter
        seasonal = (peak_load_mw - base_load_mw) * 0.4 * (summer_factor + winter_factor)
        
        # Daily pattern - different by coop type
        if coop_type == "residential":
            # Morning and evening peaks
            if 6 <= hour_of_day <= 9:
                daily = 0.15 * base_load_mw  # Morning
            elif 17 <= hour_of_day <= 21:
                daily = 0.25 * base_load_mw  # Evening peak
            elif 23 <= hour_of_day or hour_of_day <= 5:
                daily = -0.2 * base_load_mw  # Night valley
            else:
                daily = 0
                
        elif coop_type == "agricultural":
            # Irrigation typically runs overnight (cheaper) or midday
            # Strong seasonal component
            if 100 <= day_of_year <= 250:  # Growing season
                if 10 <= hour_of_day <= 16:
                    daily = 0.3 * base_load_mw
                elif 22 <= hour_of_day or hour_of_day <= 5:
                    daily = 0.2 * base_load_mw  # Night irrigation
                else:
                    daily = 0
            else:
                daily = -0.1 * base_load_mw
                
        else:  # mixed
            # Combination of patterns
            if 6 <= hour_of_day <= 9:
                daily = 0.1 * base_load_mw
            elif 14 <= hour_of_day <= 18:
                daily = 0.2 * base_load_mw  # Afternoon peak (A/C + some ag)
            elif 19 <= hour_of_day <= 21:
                daily = 0.15 * base_load_mw
            elif 23 <= hour_of_day or hour_of_day <= 5:
                daily = -0.15 * base_load_mw
            else:
                daily = 0
        
        # Weekend effect (slightly lower for commercial/ag)
        weekend_factor = 0.95 if day_of_week >= 5 else 1.0
        
        # Temperature effect on summer days
        # Simulate: hotter afternoons = more A/C
        if 150 <= day_of_year <= 250 and 14 <= hour_of_day <= 19:
            temp_spike = 0.15 * base_load_mw * np.random.uniform(0.5, 1.5)
        else:
            temp_spike = 0
        
        # Random variation (weather, random events)
        noise = np.random.normal(0, 0.05 * base_load_mw)
        
        # Combine all components
        total_load = (load + seasonal + daily + temp_spike + noise) * weekend_factor
        total_load = max(total_load, base_load_mw * 0.3)  # Floor at 30% of base
        
        loads.append({
            "datetime": hour,
            "load_mw": round(total_load, 2)
        })
    
    return pd.DataFrame(loads)


def generate_multiple_profiles() -> dict:
    """Generate load profiles for different coop archetypes."""
    profiles = {}
    
    # Small residential coop
    profiles["small_residential"] = generate_coop_load_profile(
        days=365,
        base_load_mw=25,
        peak_load_mw=60,
        coop_type="residential"
    )
    
    # Medium mixed coop (typical)
    profiles["medium_mixed"] = generate_coop_load_profile(
        days=365,
        base_load_mw=50,
        peak_load_mw=120,
        coop_type="mixed"
    )
    
    # Agricultural heavy coop
    profiles["agricultural"] = generate_coop_load_profile(
        days=365,
        base_load_mw=40,
        peak_load_mw=150,  # Big irrigation peaks
        coop_type="agricultural"
    )
    
    return profiles


def load_profile_stats(df: pd.DataFrame) -> dict:
    """Calculate key statistics for a load profile."""
    return {
        "total_mwh": df["load_mw"].sum(),  # Total annual energy
        "peak_mw": df["load_mw"].max(),
        "min_mw": df["load_mw"].min(),
        "avg_mw": df["load_mw"].mean(),
        "load_factor": df["load_mw"].mean() / df["load_mw"].max(),  # Avg/Peak
    }


def save_load_profile(df: pd.DataFrame, name: str):
    """Save load profile to parquet."""
    filepath = DATA_DIR / f"load_profile_{name}.parquet"
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath}")


def load_load_profile(name: str) -> pd.DataFrame:
    """Load profile from parquet."""
    filepath = DATA_DIR / f"load_profile_{name}.parquet"
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating coop load profiles...\n")
    
    profiles = generate_multiple_profiles()
    
    for name, df in profiles.items():
        stats = load_profile_stats(df)
        print(f"\n{name.upper()}:")
        print(f"  Total Energy: {stats['total_mwh']:,.0f} MWh/year")
        print(f"  Peak Demand:  {stats['peak_mw']:.1f} MW")
        print(f"  Average Load: {stats['avg_mw']:.1f} MW")
        print(f"  Load Factor:  {stats['load_factor']:.1%}")
        
        save_load_profile(df, name)
    
    # Also show sample daily pattern
    print("\n\nSample daily pattern (medium_mixed, summer day):")
    summer_day = profiles["medium_mixed"][
        (profiles["medium_mixed"]["datetime"].dt.month == 7) & 
        (profiles["medium_mixed"]["datetime"].dt.day == 15)
    ]
    for _, row in summer_day.iterrows():
        hour = row["datetime"].hour
        load = row["load_mw"]
        bar = "█" * int(load / 5)
        print(f"  {hour:02d}:00  {load:6.1f} MW  {bar}")
