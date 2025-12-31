"""
Hedge Analysis Module

Compare different hedging strategies:
1. No hedge (100% spot exposure)
2. Fixed swap (lock in price for all volume)
3. Partial hedge (hedge X% of expected load)
4. Collar (cap + floor)
5. Block hedges (peak vs off-peak)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class HedgeStrategy:
    """Defines a hedging strategy."""
    name: str
    description: str
    hedge_ratio: float  # 0.0 to 1.0, portion of load hedged
    fixed_price: float = None  # For swaps
    cap_price: float = None  # For collars
    floor_price: float = None  # For collars


def calculate_hedged_cost(
    load_mw: np.ndarray,
    spot_prices: np.ndarray,
    strategy: HedgeStrategy,
) -> float:
    """
    Calculate total cost under a hedging strategy.
    
    Args:
        load_mw: Hourly load in MW
        spot_prices: Hourly spot prices $/MWh
        strategy: Hedging strategy to apply
        
    Returns:
        Total cost in $
    """
    total_mwh = load_mw.sum()
    hedged_mwh = total_mwh * strategy.hedge_ratio
    unhedged_mwh = total_mwh - hedged_mwh
    
    # Unhedged portion pays spot
    # Simplified: assume unhedged portion is proportional each hour
    spot_cost = (load_mw * spot_prices * (1 - strategy.hedge_ratio)).sum()
    
    if strategy.fixed_price is not None:
        # Swap: pay fixed price for hedged volume
        hedge_cost = hedged_mwh * strategy.fixed_price
        
    elif strategy.cap_price is not None and strategy.floor_price is not None:
        # Collar: capped upside, floored downside
        capped_prices = np.clip(spot_prices, strategy.floor_price, strategy.cap_price)
        hedge_cost = (load_mw * capped_prices * strategy.hedge_ratio).sum()
        
    else:
        hedge_cost = 0
    
    return spot_cost + hedge_cost


def calculate_hedge_cost_distribution(
    load_mw: np.ndarray,
    price_simulations: np.ndarray,
    strategy: HedgeStrategy,
) -> np.ndarray:
    """
    Calculate cost distribution under a hedging strategy across simulations.
    
    Args:
        load_mw: Hourly load (n_hours,)
        price_simulations: Simulated prices (n_sims, n_hours)
        strategy: Hedging strategy
        
    Returns:
        Array of costs (n_sims,)
    """
    n_sims = price_simulations.shape[0]
    costs = np.zeros(n_sims)
    
    for i in range(n_sims):
        costs[i] = calculate_hedged_cost(
            load_mw,
            price_simulations[i],
            strategy
        )
    
    return costs


def compare_strategies(
    load_mw: np.ndarray,
    price_simulations: np.ndarray,
    current_forward_price: float,
) -> Dict[str, Dict]:
    """
    Compare multiple hedging strategies.
    
    Args:
        load_mw: Hourly load
        price_simulations: Monte Carlo price simulations
        current_forward_price: Current market forward price for swaps
        
    Returns:
        Dict with results for each strategy
    """
    # Define strategies to compare
    strategies = [
        HedgeStrategy(
            name="no_hedge",
            description="100% Spot Exposure",
            hedge_ratio=0.0,
        ),
        HedgeStrategy(
            name="full_swap",
            description=f"100% Fixed Swap @ ${current_forward_price:.2f}/MWh",
            hedge_ratio=1.0,
            fixed_price=current_forward_price,
        ),
        HedgeStrategy(
            name="partial_50",
            description=f"50% Fixed Swap @ ${current_forward_price:.2f}/MWh",
            hedge_ratio=0.5,
            fixed_price=current_forward_price,
        ),
        HedgeStrategy(
            name="partial_75",
            description=f"75% Fixed Swap @ ${current_forward_price:.2f}/MWh",
            hedge_ratio=0.75,
            fixed_price=current_forward_price,
        ),
        HedgeStrategy(
            name="collar",
            description=f"Collar: Floor ${current_forward_price*0.8:.2f}, Cap ${current_forward_price*1.2:.2f}",
            hedge_ratio=1.0,
            cap_price=current_forward_price * 1.2,
            floor_price=current_forward_price * 0.8,
        ),
    ]
    
    results = {}
    
    for strategy in strategies:
        cost_dist = calculate_hedge_cost_distribution(
            load_mw,
            price_simulations,
            strategy
        )
        
        results[strategy.name] = {
            "description": strategy.description,
            "expected_cost": np.mean(cost_dist),
            "cost_std": np.std(cost_dist),
            "cost_95_var": np.percentile(cost_dist, 95),
            "cost_5_best": np.percentile(cost_dist, 5),
            "cost_min": np.min(cost_dist),
            "cost_max": np.max(cost_dist),
            "distribution": cost_dist,
        }
    
    return results


def format_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Format strategy comparison as a table."""
    rows = []
    
    for name, data in results.items():
        rows.append({
            "Strategy": data["description"],
            "Expected Cost": f"${data['expected_cost']:,.0f}",
            "Std Dev": f"${data['cost_std']:,.0f}",
            "95% Worst": f"${data['cost_95_var']:,.0f}",
            "5% Best": f"${data['cost_5_best']:,.0f}",
            "Range": f"${data['cost_max'] - data['cost_min']:,.0f}",
        })
    
    return pd.DataFrame(rows)


def calculate_hedge_effectiveness(
    no_hedge_dist: np.ndarray,
    hedged_dist: np.ndarray,
) -> Dict:
    """
    Calculate how effective a hedge is at reducing risk.
    
    Returns metrics comparing hedged vs unhedged.
    """
    return {
        "var_reduction": 1 - (np.std(hedged_dist) / np.std(no_hedge_dist)),
        "tail_risk_reduction": 1 - (
            (np.percentile(hedged_dist, 95) - np.mean(hedged_dist)) /
            (np.percentile(no_hedge_dist, 95) - np.mean(no_hedge_dist))
        ),
        "expected_cost_change": np.mean(hedged_dist) - np.mean(no_hedge_dist),
        "worst_case_improvement": np.max(no_hedge_dist) - np.max(hedged_dist),
    }


if __name__ == "__main__":
    # Demo
    from data_fetcher import generate_synthetic_lmp
    from load_profiles import generate_coop_load_profile
    from risk_models import simulate_prices_monte_carlo
    
    print("Generating demo data...")
    lmp_df = generate_synthetic_lmp(days=365, nodes=["COOP_NODE_1"])
    load_df = generate_coop_load_profile(days=365, base_load_mw=50, peak_load_mw=100)
    
    # Get price series
    prices = lmp_df[lmp_df["node"] == "COOP_NODE_1"]["lmp"]
    
    # Current forward price (use historical average as proxy)
    forward_price = prices.mean()
    print(f"\nAssumed forward price: ${forward_price:.2f}/MWh")
    
    # Simulate prices
    print("Running Monte Carlo simulation...")
    price_sims = simulate_prices_monte_carlo(
        prices,
        hours_to_simulate=len(load_df),
        n_simulations=1000
    )
    
    # Compare strategies
    print("Comparing hedging strategies...")
    results = compare_strategies(
        load_mw=load_df["load_mw"].values,
        price_simulations=price_sims,
        current_forward_price=forward_price,
    )
    
    # Display results
    print("\n" + "="*80)
    print("HEDGING STRATEGY COMPARISON")
    print("="*80)
    
    table = format_comparison_table(results)
    print(table.to_string(index=False))
    
    # Effectiveness analysis
    print("\n" + "="*80)
    print("HEDGE EFFECTIVENESS (Full Swap vs No Hedge)")
    print("="*80)
    
    effectiveness = calculate_hedge_effectiveness(
        results["no_hedge"]["distribution"],
        results["full_swap"]["distribution"]
    )
    
    print(f"Volatility Reduction:    {effectiveness['var_reduction']:.1%}")
    print(f"Tail Risk Reduction:     {effectiveness['tail_risk_reduction']:.1%}")
    print(f"Expected Cost Change:    ${effectiveness['expected_cost_change']:+,.0f}")
    print(f"Worst Case Improvement:  ${effectiveness['worst_case_improvement']:,.0f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    no_hedge = results["no_hedge"]
    full_swap = results["full_swap"]
    
    print(f"""
Without hedging:
  - Expected cost: ${no_hedge['expected_cost']:,.0f}
  - But could range from ${no_hedge['cost_5_best']:,.0f} to ${no_hedge['cost_95_var']:,.0f}
  - That's ${no_hedge['cost_95_var'] - no_hedge['cost_5_best']:,.0f} of uncertainty

With full swap at ${forward_price:.2f}/MWh:
  - Locked in cost: ${full_swap['expected_cost']:,.0f}
  - Virtually no uncertainty

Trade-off:
  - Swap eliminates upside (prices could drop)
  - But protects against ${no_hedge['cost_95_var'] - full_swap['expected_cost']:,.0f} worst-case scenario
""")
