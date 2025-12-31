"""
Risk Models for Energy Cost Analysis

Core calculations:
1. Historical cost analysis (load × historical prices)
2. Monte Carlo simulation of future prices
3. Cost-at-Risk (CaR) calculations
4. Value-at-Risk style metrics for energy portfolios
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def calculate_historical_cost(
    load_df: pd.DataFrame,
    lmp_df: pd.DataFrame,
    node: str,
) -> pd.DataFrame:
    """
    Calculate what a coop would have paid historically.
    
    Args:
        load_df: DataFrame with datetime, load_mw
        lmp_df: DataFrame with datetime, node, lmp
        node: Which pricing node to use
        
    Returns:
        DataFrame with datetime, load_mw, lmp, hourly_cost
    """
    # Filter LMP to the specific node
    node_lmp = lmp_df[lmp_df["node"] == node][["datetime", "lmp"]].copy()
    
    # Ensure datetime columns are the same type
    load_df = load_df.copy()
    load_df["datetime"] = pd.to_datetime(load_df["datetime"])
    node_lmp["datetime"] = pd.to_datetime(node_lmp["datetime"])
    
    # Round to hour to ensure matching
    load_df["datetime"] = load_df["datetime"].dt.floor("h")
    node_lmp["datetime"] = node_lmp["datetime"].dt.floor("h")
    
    # Merge on datetime
    merged = pd.merge(
        load_df,
        node_lmp,
        on="datetime",
        how="inner"
    )
    
    # If no overlap, try matching just by position (same length assumed)
    if len(merged) == 0 and len(load_df) == len(node_lmp):
        merged = load_df.copy()
        merged["lmp"] = node_lmp["lmp"].values
    
    # Cost = Load (MW) × Price ($/MWh) × 1 hour = $ per hour
    merged["hourly_cost"] = merged["load_mw"] * merged["lmp"]
    
    return merged


def analyze_historical_costs(cost_df: pd.DataFrame) -> Dict:
    """
    Analyze historical cost patterns.
    
    Returns dict with key metrics.
    """
    # Monthly aggregation
    cost_df = cost_df.copy()
    cost_df["month"] = cost_df["datetime"].dt.to_period("M")
    monthly = cost_df.groupby("month").agg({
        "load_mw": "sum",  # Total MWh
        "hourly_cost": "sum",  # Total $
        "lmp": "mean"  # Avg price
    }).reset_index()
    monthly["avg_cost_per_mwh"] = monthly["hourly_cost"] / monthly["load_mw"]
    
    return {
        "total_cost": cost_df["hourly_cost"].sum(),
        "total_mwh": cost_df["load_mw"].sum(),
        "avg_price": cost_df["lmp"].mean(),
        "avg_cost_per_mwh": cost_df["hourly_cost"].sum() / cost_df["load_mw"].sum(),
        "max_hourly_cost": cost_df["hourly_cost"].max(),
        "monthly_costs": monthly,
        "price_std": cost_df["lmp"].std(),
        "price_95th": cost_df["lmp"].quantile(0.95),
        "price_5th": cost_df["lmp"].quantile(0.05),
    }


def fit_price_distribution(lmp_series: pd.Series) -> Dict:
    """
    Fit a distribution to historical prices for simulation.
    
    Returns parameters for price simulation.
    """
    # Log-transform for better fit (prices are often log-normal-ish)
    # But handle negative prices
    min_price = lmp_series.min()
    shifted = lmp_series - min_price + 1  # Shift to positive
    
    log_prices = np.log(shifted)
    
    return {
        "mean": lmp_series.mean(),
        "std": lmp_series.std(),
        "log_mean": log_prices.mean(),
        "log_std": log_prices.std(),
        "min_shift": min_price - 1,
        "skew": stats.skew(lmp_series),
        "kurtosis": stats.kurtosis(lmp_series),
    }


def simulate_prices_monte_carlo(
    historical_lmp: pd.Series,
    hours_to_simulate: int = 8760,  # 1 year
    n_simulations: int = 1000,
    method: str = "bootstrap"
) -> np.ndarray:
    """
    Generate Monte Carlo price simulations.
    
    Args:
        historical_lmp: Historical price series
        hours_to_simulate: Number of hours to simulate
        n_simulations: Number of simulation paths
        method: "bootstrap" (resample history) or "parametric" (fit distribution)
        
    Returns:
        Array of shape (n_simulations, hours_to_simulate)
    """
    np.random.seed(42)
    
    if method == "bootstrap":
        # Simple bootstrap: randomly sample from historical prices
        # Preserves actual distribution including spikes
        simulations = np.random.choice(
            historical_lmp.values,
            size=(n_simulations, hours_to_simulate),
            replace=True
        )
        
    elif method == "parametric":
        # Fit distribution and sample
        # Using mixture: normal for base + exponential for spikes
        params = fit_price_distribution(historical_lmp)
        
        # Base prices (normal)
        base = np.random.normal(
            params["mean"],
            params["std"],
            size=(n_simulations, hours_to_simulate)
        )
        
        # Occasional spikes
        spike_prob = 0.01  # 1% of hours have spikes
        spikes = np.random.exponential(
            params["std"] * 2,
            size=(n_simulations, hours_to_simulate)
        )
        spike_mask = np.random.random((n_simulations, hours_to_simulate)) < spike_prob
        
        simulations = base + (spikes * spike_mask)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return simulations


def calculate_cost_distribution(
    price_simulations: np.ndarray,
    load_profile: np.ndarray,
) -> np.ndarray:
    """
    Calculate cost distribution from price simulations and load.
    
    Args:
        price_simulations: Shape (n_sims, n_hours)
        load_profile: Shape (n_hours,) - MW per hour
        
    Returns:
        Array of total costs, shape (n_sims,)
    """
    # Broadcast load across simulations
    # Cost = sum(price × load) for each simulation
    hourly_costs = price_simulations * load_profile  # Broadcasting
    total_costs = hourly_costs.sum(axis=1)  # Sum across hours
    
    return total_costs


def calculate_risk_metrics(cost_distribution: np.ndarray) -> Dict:
    """
    Calculate risk metrics from cost distribution.
    
    Returns:
        Dict with expected cost, VaR, CVaR, etc.
    """
    return {
        "expected_cost": np.mean(cost_distribution),
        "cost_std": np.std(cost_distribution),
        "cost_95_var": np.percentile(cost_distribution, 95),  # 95% VaR
        "cost_99_var": np.percentile(cost_distribution, 99),  # 99% VaR
        "cost_5_best": np.percentile(cost_distribution, 5),   # Best 5% case
        "cost_median": np.median(cost_distribution),
        # CVaR (expected shortfall) - average of worst 5%
        "cost_95_cvar": np.mean(cost_distribution[cost_distribution >= np.percentile(cost_distribution, 95)]),
        "cost_min": np.min(cost_distribution),
        "cost_max": np.max(cost_distribution),
    }


def run_risk_analysis(
    load_df: pd.DataFrame,
    lmp_df: pd.DataFrame,
    node: str,
    n_simulations: int = 1000,
) -> Dict:
    """
    Run complete risk analysis for a coop.
    
    Args:
        load_df: Load profile with datetime, load_mw
        lmp_df: LMP data with datetime, node, lmp
        node: Pricing node
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        Dict with historical analysis, simulations, and risk metrics
    """
    # 1. Historical analysis
    historical_costs = calculate_historical_cost(load_df, lmp_df, node)
    historical_analysis = analyze_historical_costs(historical_costs)
    
    # 2. Get price series for the node
    node_prices = lmp_df[lmp_df["node"] == node]["lmp"]
    
    # 3. Monte Carlo simulation
    n_hours = len(load_df)
    price_sims = simulate_prices_monte_carlo(
        node_prices,
        hours_to_simulate=n_hours,
        n_simulations=n_simulations,
        method="bootstrap"
    )
    
    # 4. Calculate cost distribution
    load_array = load_df["load_mw"].values
    cost_dist = calculate_cost_distribution(price_sims, load_array)
    
    # 5. Risk metrics
    risk_metrics = calculate_risk_metrics(cost_dist)
    
    return {
        "historical": historical_analysis,
        "risk_metrics": risk_metrics,
        "cost_distribution": cost_dist,
        "price_simulations": price_sims,  # May want to drop for memory
    }


if __name__ == "__main__":
    # Demo with synthetic data
    from data_fetcher import generate_synthetic_lmp
    from load_profiles import generate_coop_load_profile
    
    print("Generating demo data...")
    lmp_df = generate_synthetic_lmp(days=365, nodes=["COOP_NODE_1"])
    load_df = generate_coop_load_profile(days=365, base_load_mw=50, peak_load_mw=100)
    
    print("\nRunning risk analysis...")
    results = run_risk_analysis(
        load_df=load_df,
        lmp_df=lmp_df,
        node="COOP_NODE_1",
        n_simulations=1000
    )
    
    print("\n" + "="*60)
    print("HISTORICAL ANALYSIS")
    print("="*60)
    h = results["historical"]
    print(f"Total Cost:        ${h['total_cost']:,.0f}")
    print(f"Total Energy:      {h['total_mwh']:,.0f} MWh")
    print(f"Avg Price:         ${h['avg_price']:.2f}/MWh")
    print(f"Avg Cost/MWh:      ${h['avg_cost_per_mwh']:.2f}")
    print(f"Price Volatility:  ${h['price_std']:.2f} (std dev)")
    print(f"Price Range:       ${h['price_5th']:.2f} - ${h['price_95th']:.2f} (5th-95th)")
    
    print("\n" + "="*60)
    print("RISK ANALYSIS (Monte Carlo)")
    print("="*60)
    r = results["risk_metrics"]
    print(f"Expected Cost:     ${r['expected_cost']:,.0f}")
    print(f"Cost Std Dev:      ${r['cost_std']:,.0f}")
    print(f"95% Cost-at-Risk:  ${r['cost_95_var']:,.0f}")
    print(f"99% Cost-at-Risk:  ${r['cost_99_var']:,.0f}")
    print(f"Best 5% Case:      ${r['cost_5_best']:,.0f}")
    print(f"Cost Range:        ${r['cost_min']:,.0f} - ${r['cost_max']:,.0f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print(f"There's a 5% chance annual costs exceed ${r['cost_95_var']:,.0f}")
    print(f"Expected cost is ${r['expected_cost']:,.0f} ± ${r['cost_std']:,.0f}")
    spread = r['cost_95_var'] - r['cost_5_best']
    print(f"90% confidence range is ${spread:,.0f} wide")
