"""
Coop Energy Risk Dashboard

A demo dashboard showing energy price risk analysis for electric cooperatives.

Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import generate_synthetic_lmp, DATA_DIR
from load_profiles import generate_coop_load_profile, load_profile_stats
from risk_models import (
    calculate_historical_cost,
    analyze_historical_costs,
    simulate_prices_monte_carlo,
    calculate_cost_distribution,
    calculate_risk_metrics,
)
from hedge_analysis import compare_strategies, format_comparison_table

# Page config
st.set_page_config(
    page_title="Coop Energy Risk Demo",
    page_icon="⚡",
    layout="wide",
)

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_generate_lmp_data():
    """Load LMP data from file or generate synthetic."""
    try:
        return pd.read_parquet(DATA_DIR / "synthetic_lmp_hourly.parquet")
    except:
        return generate_synthetic_lmp(
            days=365,
            nodes=["COOP_NODE_1", "COOP_NODE_2", "COOP_NODE_3", "HUB_SPP"]
        )


@st.cache_data
def generate_load_profile(base_load: float, peak_load: float, coop_type: str):
    """Generate load profile with given parameters."""
    return generate_coop_load_profile(
        days=365,
        base_load_mw=base_load,
        peak_load_mw=peak_load,
        coop_type=coop_type,
    )


# ============================================================
# SIDEBAR - Configuration
# ============================================================

st.sidebar.title("⚡ Configuration")

st.sidebar.header("Coop Profile")
coop_type = st.sidebar.selectbox(
    "Coop Type",
    ["mixed", "residential", "agricultural"],
    help="Affects load shape pattern"
)

base_load = st.sidebar.slider(
    "Base Load (MW)",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
)

peak_load = st.sidebar.slider(
    "Peak Load (MW)",
    min_value=base_load,
    max_value=400,
    value=min(100, base_load * 2),
    step=10,
)

st.sidebar.header("Analysis Settings")
selected_node = st.sidebar.selectbox(
    "Pricing Node",
    ["COOP_NODE_1", "COOP_NODE_2", "COOP_NODE_3", "HUB_SPP"],
)

n_simulations = st.sidebar.slider(
    "Monte Carlo Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

forward_price_premium = st.sidebar.slider(
    "Forward Price Premium (%)",
    min_value=-20,
    max_value=20,
    value=5,
    help="Premium over historical average for forward/swap pricing"
)


# ============================================================
# MAIN CONTENT
# ============================================================

st.title("⚡ Coop Energy Risk Analysis")
st.markdown("*A demo platform for electric cooperative energy cost risk management*")

# Load data
lmp_df = load_or_generate_lmp_data()
load_df = generate_load_profile(base_load, peak_load, coop_type)

# ============================================================
# TAB 1: LMP Explorer
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Explorer",
    "🔌 Load Profile", 
    "📈 Risk Analysis",
    "🛡️ Hedge Comparison"
])

with tab1:
    st.header("Locational Marginal Price (LMP) Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price time series
        node_data = lmp_df[lmp_df["node"] == selected_node].copy()
        
        fig = px.line(
            node_data,
            x="datetime",
            y="lmp",
            title=f"LMP at {selected_node}",
            labels={"lmp": "Price ($/MWh)", "datetime": "Date"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price statistics
        st.subheader("Price Statistics")
        
        avg_price = node_data["lmp"].mean()
        std_price = node_data["lmp"].std()
        max_price = node_data["lmp"].max()
        min_price = node_data["lmp"].min()
        
        st.metric("Average Price", f"${avg_price:.2f}/MWh")
        st.metric("Volatility (Std Dev)", f"${std_price:.2f}")
        st.metric("Max Price", f"${max_price:.2f}/MWh")
        st.metric("Min Price", f"${min_price:.2f}/MWh")
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            node_data,
            x="lmp",
            nbins=50,
            title="Price Distribution",
            labels={"lmp": "Price ($/MWh)"}
        )
        fig.add_vline(x=avg_price, line_dash="dash", line_color="red",
                      annotation_text=f"Avg: ${avg_price:.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly averages
        node_data["month"] = node_data["datetime"].dt.to_period("M").astype(str)
        monthly = node_data.groupby("month")["lmp"].agg(["mean", "std"]).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month"],
            y=monthly["mean"],
            error_y=dict(type="data", array=monthly["std"]),
            name="Monthly Avg ± Std"
        ))
        fig.update_layout(title="Monthly Price Averages", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Node comparison
    st.subheader("Node Comparison")
    
    node_stats = lmp_df.groupby("node")["lmp"].agg(["mean", "std", "min", "max"]).reset_index()
    node_stats.columns = ["Node", "Avg Price", "Volatility", "Min", "Max"]
    node_stats = node_stats.round(2)
    st.dataframe(node_stats, use_container_width=True)


# ============================================================
# TAB 2: Load Profile
# ============================================================

with tab2:
    st.header("Coop Load Profile")
    
    stats = load_profile_stats(load_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Energy", f"{stats['total_mwh']:,.0f} MWh")
    with col2:
        st.metric("Peak Demand", f"{stats['peak_mw']:.1f} MW")
    with col3:
        st.metric("Average Load", f"{stats['avg_mw']:.1f} MW")
    with col4:
        st.metric("Load Factor", f"{stats['load_factor']:.1%}")
    
    # Load time series
    fig = px.line(
        load_df,
        x="datetime",
        y="load_mw",
        title="Annual Load Profile",
        labels={"load_mw": "Load (MW)", "datetime": "Date"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily patterns
    col1, col2 = st.columns(2)
    
    with col1:
        load_df_copy = load_df.copy()
        load_df_copy["hour"] = load_df_copy["datetime"].dt.hour
        hourly_avg = load_df_copy.groupby("hour")["load_mw"].mean().reset_index()
        
        fig = px.bar(
            hourly_avg,
            x="hour",
            y="load_mw",
            title="Average Hourly Load Pattern",
            labels={"load_mw": "Avg Load (MW)", "hour": "Hour of Day"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        load_df_copy["month"] = load_df_copy["datetime"].dt.month
        monthly_avg = load_df_copy.groupby("month")["load_mw"].agg(["mean", "max"]).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_avg["month"], y=monthly_avg["mean"], name="Average"))
        fig.add_trace(go.Scatter(x=monthly_avg["month"], y=monthly_avg["max"], 
                                  mode="lines+markers", name="Peak"))
        fig.update_layout(title="Monthly Load Pattern", xaxis_title="Month", yaxis_title="Load (MW)")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3: Risk Analysis
# ============================================================

with tab3:
    st.header("Energy Cost Risk Analysis")
    
    # Historical cost analysis
    st.subheader("Historical Cost Analysis")
    
    cost_df = calculate_historical_cost(load_df, lmp_df, selected_node)
    historical = analyze_historical_costs(cost_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Annual Cost", f"${historical['total_cost']:,.0f}")
    with col2:
        st.metric("Avg Cost/MWh", f"${historical['avg_cost_per_mwh']:.2f}")
    with col3:
        st.metric("Total Energy", f"{historical['total_mwh']:,.0f} MWh")
    with col4:
        st.metric("Price Volatility", f"${historical['price_std']:.2f}")
    
    # Monte Carlo simulation
    st.subheader("Monte Carlo Risk Simulation")
    
    with st.spinner(f"Running {n_simulations} simulations..."):
        node_prices = lmp_df[lmp_df["node"] == selected_node]["lmp"]
        
        price_sims = simulate_prices_monte_carlo(
            node_prices,
            hours_to_simulate=len(load_df),
            n_simulations=n_simulations,
            method="bootstrap"
        )
        
        cost_dist = calculate_cost_distribution(
            price_sims,
            load_df["load_mw"].values
        )
        
        risk_metrics = calculate_risk_metrics(cost_dist)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Expected Annual Cost",
            f"${risk_metrics['expected_cost']:,.0f}",
            help="Average across all simulations"
        )
    with col2:
        st.metric(
            "95% Cost-at-Risk",
            f"${risk_metrics['cost_95_var']:,.0f}",
            delta=f"+${risk_metrics['cost_95_var'] - risk_metrics['expected_cost']:,.0f}",
            delta_color="inverse",
            help="5% chance costs exceed this"
        )
    with col3:
        st.metric(
            "Cost Uncertainty",
            f"±${risk_metrics['cost_std']:,.0f}",
            help="Standard deviation of cost"
        )
    
    # Cost distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            x=cost_dist,
            nbins=50,
            title="Simulated Annual Cost Distribution",
            labels={"x": "Annual Cost ($)"}
        )
        fig.add_vline(x=risk_metrics['expected_cost'], line_dash="solid", line_color="green",
                      annotation_text="Expected")
        fig.add_vline(x=risk_metrics['cost_95_var'], line_dash="dash", line_color="red",
                      annotation_text="95% VaR")
        fig.add_vline(x=risk_metrics['cost_5_best'], line_dash="dash", line_color="blue",
                      annotation_text="5% Best")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics summary
        st.markdown("### Risk Summary")
        
        range_90 = risk_metrics['cost_95_var'] - risk_metrics['cost_5_best']
        pct_range = range_90 / risk_metrics['expected_cost'] * 100
        
        st.markdown(f"""
        Based on {n_simulations:,} Monte Carlo simulations:
        
        - **Expected Cost:** ${risk_metrics['expected_cost']:,.0f}
        - **Best Case (5%):** ${risk_metrics['cost_5_best']:,.0f}
        - **Worst Case (95%):** ${risk_metrics['cost_95_var']:,.0f}
        - **90% Confidence Range:** ${range_90:,.0f} ({pct_range:.1f}% of expected)
        
        **Interpretation:** There's a 5% chance annual energy costs 
        could exceed ${risk_metrics['cost_95_var']:,.0f}, which is 
        ${risk_metrics['cost_95_var'] - risk_metrics['expected_cost']:,.0f} 
        more than expected.
        """)


# ============================================================
# TAB 4: Hedge Comparison
# ============================================================

with tab4:
    st.header("Hedging Strategy Comparison")
    
    # Calculate forward price (historical avg + premium)
    historical_avg = lmp_df[lmp_df["node"] == selected_node]["lmp"].mean()
    forward_price = historical_avg * (1 + forward_price_premium / 100)
    
    st.info(f"""
    **Forward Price Assumption:** ${forward_price:.2f}/MWh 
    (Historical average ${historical_avg:.2f} + {forward_price_premium}% premium)
    """)
    
    with st.spinner("Analyzing hedging strategies..."):
        results = compare_strategies(
            load_mw=load_df["load_mw"].values,
            price_simulations=price_sims,
            current_forward_price=forward_price,
        )
    
    # Strategy comparison table
    st.subheader("Strategy Comparison")
    
    comparison_data = []
    for name, data in results.items():
        comparison_data.append({
            "Strategy": data["description"],
            "Expected Cost": data["expected_cost"],
            "Std Dev": data["cost_std"],
            "95% Worst Case": data["cost_95_var"],
            "5% Best Case": data["cost_5_best"],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format for display
    st.dataframe(
        comparison_df.style.format({
            "Expected Cost": "${:,.0f}",
            "Std Dev": "${:,.0f}",
            "95% Worst Case": "${:,.0f}",
            "5% Best Case": "${:,.0f}",
        }),
        use_container_width=True
    )
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for name, data in results.items():
            fig.add_trace(go.Box(
                y=data["distribution"],
                name=name.replace("_", " ").title(),
                boxmean=True,
            ))
        fig.update_layout(
            title="Cost Distribution by Strategy",
            yaxis_title="Annual Cost ($)",
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk vs Cost scatter
        scatter_data = pd.DataFrame([
            {
                "Strategy": name.replace("_", " ").title(),
                "Expected Cost": data["expected_cost"],
                "Risk (Std Dev)": data["cost_std"],
            }
            for name, data in results.items()
        ])
        
        fig = px.scatter(
            scatter_data,
            x="Risk (Std Dev)",
            y="Expected Cost",
            text="Strategy",
            title="Risk vs Expected Cost",
            size_max=60,
        )
        fig.update_traces(textposition="top center", marker=dict(size=15))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Hedge effectiveness
    st.subheader("Hedge Effectiveness Analysis")
    
    no_hedge = results["no_hedge"]
    full_swap = results["full_swap"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vol_reduction = 1 - (full_swap["cost_std"] / no_hedge["cost_std"])
        st.metric(
            "Volatility Reduction",
            f"{vol_reduction:.1%}",
            help="How much the swap reduces cost uncertainty"
        )
    
    with col2:
        worst_case_improvement = no_hedge["cost_95_var"] - full_swap["cost_95_var"]
        st.metric(
            "Worst Case Savings",
            f"${worst_case_improvement:,.0f}",
            help="Reduction in 95% worst-case cost"
        )
    
    with col3:
        cost_change = full_swap["expected_cost"] - no_hedge["expected_cost"]
        st.metric(
            "Expected Cost Change",
            f"${cost_change:+,.0f}",
            delta_color="inverse" if cost_change > 0 else "normal",
            help="Cost of hedging (premium paid)"
        )
    
    st.markdown(f"""
    ### Key Insights
    
    **Without hedging:**
    - Expected annual cost: ${no_hedge['expected_cost']:,.0f}
    - Could range from ${no_hedge['cost_5_best']:,.0f} to ${no_hedge['cost_95_var']:,.0f}
    - That's ${no_hedge['cost_95_var'] - no_hedge['cost_5_best']:,.0f} of budget uncertainty
    
    **With a full swap at ${forward_price:.2f}/MWh:**
    - Locked-in cost: ~${full_swap['expected_cost']:,.0f}
    - Eliminates {vol_reduction:.0%} of cost volatility
    - Protects against ${worst_case_improvement:,.0f} worst-case scenario
    
    **Trade-off:**
    - The swap costs ~${cost_change:,.0f} more than expected spot
    - But eliminates budget risk for board/rate planning
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p><strong>Coop Energy Risk Demo</strong> | Built for demonstration purposes</p>
    <p>Data is synthetic. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
