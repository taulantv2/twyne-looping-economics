"""
Twyne stETH-ETH Looping Economics - Interactive Dashboard

A Streamlit app for exploring the economics of leveraged stETH-ETH yield farming
using the Twyne credit delegation protocol.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from twyne_looping import (
    load_apr_data, compute_spreads, compute_rolling_stats,
    get_summary_stats,
    psi, interest_rate, looped_yield_from_leverage,
    leverage_from_ltv, ltv_from_leverage, ltv_from_hf,
    days_to_liquidation, net_rate, hf_decay_factor,
    DEFAULTS,
)
from twyne_looping.economics import compute_yield_grid, compute_days_to_liq_grid
from twyne_looping.backtest import run_backtest, run_multi_leverage_backtest
from twyne_looping.live_rates import fetch_live_rates, LiveRates, AVAILABLE_CHAINS

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Twyne stETH-ETH Looping",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Load Data
# =============================================================================

@st.cache_data
def load_data():
    """Load and preprocess historical APR data."""
    data_path = Path(__file__).parent / "data" / "stETH-ETH_historical_APR_data.csv"
    df = load_apr_data(data_path)
    df = compute_spreads(df)
    df = compute_rolling_stats(df, windows=[7, 30, 90])
    return df

df = load_data()

# =============================================================================
# Sidebar - Parameter Controls
# =============================================================================

# =============================================================================
# External Market Liquidation Thresholds
# =============================================================================

# Aave V3 e-mode liquidation thresholds by chain and asset (ETH-correlated e-mode)
AAVE_LIQ_THRESHOLDS = {
    "Ethereum": {"wstETH": 0.93, "WETH": 0.93},   # e-mode ID 1 (ETH correlated)
    "Arbitrum": {"wstETH": 0.93, "WETH": 0.93},   # e-mode available
    "Optimism": {"wstETH": 0.93, "WETH": 0.93},   # e-mode available
    "Base": {"wstETH": 0.93, "WETH": 0.93},       # e-mode available
    "Polygon": {"wstETH": 0.75, "WETH": 0.825},   # No stETH e-mode, use standard
}

# Euler V2 liquidation thresholds by chain and asset
EULER_LIQ_THRESHOLDS = {
    "Ethereum": {"wstETH": 0.90, "WETH": 0.90},   # Euler Prime vaults
    "Arbitrum": {"wstETH": 0.85, "WETH": 0.85},
    "Optimism": {"wstETH": 0.85, "WETH": 0.85},
    "Base": {"wstETH": 0.85, "WETH": 0.85},
    "Polygon": {"wstETH": 0.80, "WETH": 0.80},
}

# Map market names to threshold dictionaries
MARKET_LIQ_THRESHOLDS = {
    "Aave": AAVE_LIQ_THRESHOLDS,
    "Euler": EULER_LIQ_THRESHOLDS,
}

st.sidebar.title("⚙️ Parameters")

# =============================================================================
# Position Parameters (at top)
# =============================================================================
st.sidebar.subheader("Position")

# Chain selection
selected_chain = st.sidebar.selectbox(
    "Chain",
    options=AVAILABLE_CHAINS,
    index=0,
    help="Select blockchain for rates and liquidation parameters"
)

# External lending market selection
lending_market = st.sidebar.selectbox(
    "Lending Market",
    options=["Aave", "Euler"],
    index=0,
    help="External lending market for the looping strategy"
)

# CLP asset selection
clp_asset = st.sidebar.selectbox(
    "CLP Asset",
    options=["wstETH", "WETH"],
    index=0,
    help="Asset used as collateral by the Credit Liquidity Provider"
)

# Display C asset (always stETH for this strategy)
st.sidebar.caption("**Collateral (C):** stETH")

# Get liquidation thresholds based on selected market
market_thresholds = MARKET_LIQ_THRESHOLDS.get(lending_market, AAVE_LIQ_THRESHOLDS)
default_liq_ltv_c = market_thresholds.get(selected_chain, {}).get("wstETH", 0.90)
default_liq_ltv_clp = market_thresholds.get(selected_chain, {}).get(clp_asset, 0.90)

# Track changes to reset slider values
if 'prev_clp_asset' not in st.session_state:
    st.session_state.prev_clp_asset = clp_asset
if 'prev_chain' not in st.session_state:
    st.session_state.prev_chain = selected_chain
if 'prev_lending_market' not in st.session_state:
    st.session_state.prev_lending_market = lending_market

# Reset external market sliders when chain, CLP asset, or lending market changes
if (st.session_state.prev_clp_asset != clp_asset or
    st.session_state.prev_chain != selected_chain or
    st.session_state.prev_lending_market != lending_market):
    st.session_state.ext_liq_ltv_clp = int(default_liq_ltv_clp * 100)
    st.session_state.ext_liq_ltv_c = int(default_liq_ltv_c * 100)
    st.session_state.prev_clp_asset = clp_asset
    st.session_state.prev_chain = selected_chain
    st.session_state.prev_lending_market = lending_market

# HF slider
initial_hf = st.sidebar.slider(
    "Health Factor (HF)",
    min_value=1.01, max_value=1.20, value=1.05, step=0.01,
    help="Health Factor = λ̃_t / λ_t"
)

# Get values from session state for leverage bounds calculation (set by accordions below)
liq_ltv_c = st.session_state.get("ext_liq_ltv_c", int(default_liq_ltv_c * 100)) / 100
liq_ltv_max = st.session_state.get("twyne_liq_ltv_max", 98) / 100

# Calculate leverage bounds
# L_min = 1/(1 - λ̃_e^C) - minimum leverage (Aave's native max)
# L_max = 1 / [1 - λ̃_t^max/HF] - maximum leverage given HF and max boosted LTV
L_min = 1.0 / (1.0 - liq_ltv_c)
L_max_raw = 1.0 / (1.0 - liq_ltv_max / initial_hf) if initial_hf > liq_ltv_max else 100.0
L_max = min(max(L_max_raw, L_min + 1), 100.0)  # Ensure valid range

# Leverage slider (in Position section, after HF)
leverage_slider = st.sidebar.slider(
    "Leverage (L)",
    min_value=float(L_min),
    max_value=float(L_max),
    value=float(min(max(20.0, L_min), L_max)),
    step=1.0,
    help=f"Range: {L_min:.1f}x to {L_max:.1f}x (based on HF={initial_hf:.2f} and λ̃_t^max={liq_ltv_max*100:.0f}%)"
)
# Clamp leverage if L_max decreased below current selection
leverage = min(leverage_slider, L_max)

# =============================================================================
# Rate Source (right after Position)
# =============================================================================
st.sidebar.subheader("Rate Source")
rate_source = st.sidebar.radio(
    "Use rates from:",
    ["Current", "Custom"],
    index=0,
    horizontal=True,
    help="'Current' fetches live rates from Aave and Lido APIs"
)

# Initialize session state for live rates
if 'live_rates' not in st.session_state:
    st.session_state.live_rates = None
if 'live_rates_chain' not in st.session_state:
    st.session_state.live_rates_chain = selected_chain

if rate_source == "Current":
    # Refresh button
    if st.sidebar.button("🔄 Fetch Live Rates", width='stretch'):
        with st.spinner("Fetching rates..."):
            st.session_state.live_rates = fetch_live_rates(selected_chain)
            st.session_state.live_rates_chain = selected_chain

    # Auto-fetch if no rates cached or chain changed
    if st.session_state.live_rates is None or st.session_state.live_rates_chain != selected_chain:
        with st.spinner("Fetching rates..."):
            st.session_state.live_rates = fetch_live_rates(selected_chain)
            st.session_state.live_rates_chain = selected_chain

    # Display rates
    if st.session_state.live_rates is not None:
        live = st.session_state.live_rates
        r_steth_stake = live.steth_staking_rate
        r_steth_lend = live.steth_supply_rate
        r_stake = live.r_stake  # staking + lending
        r_borrow = live.eth_borrow_rate

        st.sidebar.caption(f"• stETH Staking (Lido): {r_steth_stake*100:.2f}%")
        st.sidebar.caption(f"• stETH Lending (Aave): {r_steth_lend*100:.2f}%")
        st.sidebar.caption(f"• ETH Borrow (Aave): {r_borrow*100:.2f}%")
    else:
        st.sidebar.error("Failed to fetch live rates. Using fallback.")
        r_stake = 0.035 + 0.001
        r_borrow = 0.025
else:
    # Custom mode - user inputs rates manually
    r_steth_stake = st.sidebar.slider(
        "stETH Staking Rate (Lido) (%)",
        min_value=0.0, max_value=10.0, value=3.5, step=0.1,
        help="Lido stETH staking APR"
    ) / 100
    r_steth_lend = st.sidebar.slider(
        "stETH Lending Rate (Aave) (%)",
        min_value=0.0, max_value=5.0, value=0.1, step=0.05,
        help="Aave stETH supply/lending APR"
    ) / 100
    r_stake = r_steth_stake + r_steth_lend
    st.sidebar.caption(f"r_stake = {r_stake*100:.2f}%")

    r_borrow = st.sidebar.slider(
        "ETH Borrow Rate (%)",
        min_value=0.0, max_value=25.0, value=2.5, step=0.1,
        help="Aave ETH borrow rate"
    ) / 100

# =============================================================================
# Raw Spread and Net Yield Metrics (right after Rate Source)
# =============================================================================
# Get params from session state for early calculation (before accordions render)
_liq_ltv_c = st.session_state.get("ext_liq_ltv_c", int(default_liq_ltv_c * 100)) / 100
_liq_ltv_clp = st.session_state.get("ext_liq_ltv_clp", int(default_liq_ltv_clp * 100)) / 100
_utilization = st.session_state.get("twyne_utilization", 95) / 100
_beta_safe = st.session_state.get("twyne_beta_safe", 1.00)
_ir_I_min = st.session_state.get("ir_I_min", 0.0) / 100
_ir_I_0 = st.session_state.get("ir_I_0", 0.8) / 100
_ir_u_0 = st.session_state.get("ir_u_0", 90) / 100
_ir_I_max = st.session_state.get("ir_I_max", 20.0) / 100
_ir_gamma = st.session_state.get("ir_gamma", 32)

# Calculate metrics for display
_ir_u = interest_rate(_utilization, I_min=_ir_I_min, I_0=_ir_I_0, u_0=_ir_u_0, I_max=_ir_I_max, gamma=_ir_gamma)
_lambda_t = ltv_from_leverage(leverage)
_liq_ltv_t = _lambda_t * initial_hf
_psi_val = psi(_liq_ltv_t, _beta_safe, _liq_ltv_clp, _liq_ltv_c)
_current_yield = looped_yield_from_leverage(r_stake, r_borrow, leverage, _ir_u, _psi_val)

# Display Spread and Yield in compact format
_raw_spread = (r_stake - r_borrow) * 100
st.sidebar.markdown(
    f"**Spread:** {_raw_spread:.2f}% &nbsp;&nbsp; **Yield:** {_current_yield*100:.2f}%",
    help="Spread = r_stake - r_borrow; Yield = net yield at current leverage"
)

# =============================================================================
# External Market Parameters (accordion)
# =============================================================================
# Dynamic label based on selected lending market
_emode_note = " (e-mode)" if lending_market == "Aave" else ""
with st.sidebar.expander(f"External Market ({lending_market}{_emode_note}) - Advanced"):
    st.caption(f"Defaults for {selected_chain} with C=stETH, CLP={clp_asset}")

    liq_ltv_c = st.slider(
        "Collateral Liq. LTV (λ̃_e^C) (%)",
        min_value=70, max_value=96, value=int(default_liq_ltv_c * 100), step=1,
        help=f"Liquidation threshold for stETH on {lending_market}",
        key="ext_liq_ltv_c"
    ) / 100

    liq_ltv_clp = st.slider(
        "CLP Collateral Liq. LTV (λ̃_e^CLP) (%)",
        min_value=70, max_value=96, value=int(default_liq_ltv_clp * 100), step=1,
        help=f"Liquidation threshold for {clp_asset} on {lending_market}",
        key="ext_liq_ltv_clp"
    ) / 100

# Get liq_ltv_clp from session state (set by accordion above)
liq_ltv_clp = st.session_state.get("ext_liq_ltv_clp", int(default_liq_ltv_clp * 100)) / 100

# =============================================================================
# Twyne Parameters (accordion with IR model sub-section)
# =============================================================================
with st.sidebar.expander("Twyne Parameters - Advanced"):
    liq_ltv_max = st.slider(
        "Max Boosted Liq. LTV (λ̃_t^max) (%)",
        min_value=90, max_value=99, value=98, step=1,
        help="Maximum allowed boosted liquidation LTV",
        key="twyne_liq_ltv_max"
    ) / 100

    utilization = st.slider(
        "CLP Utilization (u) (%)",
        min_value=0, max_value=100, value=95, step=5,
        help="Credit LP pool utilization - affects the interest rate IR(u)",
        key="twyne_utilization"
    ) / 100

    beta_safe = st.slider(
        "Safety Buffer (β_safe)",
        min_value=0.85, max_value=1.00, value=1.00, step=0.01,
        help="Twyne safety buffer parameter from whitepaper Eq. 7",
        key="twyne_beta_safe"
    )

    # Interest Rate Model sub-section
    st.markdown("---")
    st.caption("**Interest Rate Model**")
    ir_I_min = st.slider("I_min (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="ir_I_min") / 100
    ir_I_0 = st.slider("I_0 (%)", min_value=0.0, max_value=10.0, value=0.8, step=0.1, key="ir_I_0") / 100
    ir_u_0 = st.slider("u_0 (%)", min_value=50, max_value=99, value=90, step=1, key="ir_u_0") / 100
    ir_I_max = st.slider("I_max (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0, key="ir_I_max") / 100
    ir_gamma = st.slider("γ (curvature)", min_value=1, max_value=64, value=32, step=1, key="ir_gamma")

# Get Twyne params from session state
liq_ltv_max = st.session_state.get("twyne_liq_ltv_max", 98) / 100
utilization = st.session_state.get("twyne_utilization", 95) / 100
beta_safe = st.session_state.get("twyne_beta_safe", 1.00)
ir_I_min = st.session_state.get("ir_I_min", 0.0) / 100
ir_I_0 = st.session_state.get("ir_I_0", 0.8) / 100
ir_u_0 = st.session_state.get("ir_u_0", 90) / 100
ir_I_max = st.session_state.get("ir_I_max", 20.0) / 100
ir_gamma = st.session_state.get("ir_gamma", 32)

# Store parameters for use in calculations
params = {
    'beta_safe': beta_safe,
    'liq_ltv_c': liq_ltv_c,
    'liq_ltv_clp': liq_ltv_clp,
    'I_min': ir_I_min,
    'I_0': ir_I_0,
    'u_0': ir_u_0,
    'I_max': ir_I_max,
    'gamma': ir_gamma,
}

# Calculate interest rate (used across multiple tabs)
ir_u = interest_rate(utilization, **{k: params[k] for k in ['I_min', 'I_0', 'u_0', 'I_max', 'gamma']})


# =============================================================================
# Main Content - Tabs
# =============================================================================

st.title("🔄 Twyne stETH-ETH Looping Economics")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Detailed Sheet",
    "🧮 Yield Calculator",
    "📈 Historical Analysis",
    "⏱️ Backtest"
])

# =============================================================================
# Tab 1: Detailed Sheet (Heatmaps)
# =============================================================================

with tab1:
    st.header("Detailed Sheet: Yield & Risk Matrices")

    st.markdown("""
    These heatmaps reproduce the **Detailed Sheet** from the economics spreadsheet.
    Adjust parameters in the sidebar to see how yields and liquidation timelines change.
    """)

    # Grid parameters
    col1, col2 = st.columns(2)
    with col1:
        hf_min = st.number_input("HF Min", value=1.00, min_value=1.00, max_value=1.50, step=0.01)
        hf_max = st.number_input("HF Max", value=1.10, min_value=1.01, max_value=2.00, step=0.01)
    with col2:
        ltv_min = st.number_input("Liq LTV Min (%)", value=95.0, min_value=80.0, max_value=99.0, step=0.5) / 100
        ltv_max = st.number_input("Liq LTV Max (%)", value=98.0, min_value=80.0, max_value=99.0, step=0.5) / 100

    n_hf, n_ltv = 15, 10
    hf_values = np.linspace(hf_max, hf_min, n_hf)  # High to low
    liq_ltv_values = np.linspace(ltv_min, ltv_max, n_ltv)

    # Compute grids
    yield_grid = compute_yield_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                     utilization=utilization, **params)
    dtl_grid = compute_days_to_liq_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                         utilization=utilization, **params)

    # Current position from sidebar sliders
    # λ_t = (L-1)/L, and λ̃_t = HF * λ_t = HF * (L-1) / L
    current_liq_ltv = initial_hf * (leverage - 1) / leverage

    # Create heatmaps
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Net Yield (%)")

        fig_yield = go.Figure(data=go.Heatmap(
            z=yield_grid * 100,
            x=[f"{v*100:.1f}%" for v in liq_ltv_values],
            y=[f"{v:.2f}" for v in hf_values],
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Yield (%)"),
            hovertemplate="Liq LTV: %{x}<br>HF: %{y}<br>Yield: %{z:.2f}%<extra></extra>",
        ))

        # Add star marker for current position
        fig_yield.add_trace(go.Scatter(
            x=[f"{current_liq_ltv*100:.1f}%"],
            y=[f"{initial_hf:.2f}"],
            mode='markers',
            marker=dict(size=18, color='#F18F01', symbol='star', line=dict(width=2, color='black')),
            name=f'Current ({leverage:.0f}x)',
            hovertemplate=f"Current Position<br>L={leverage:.0f}x, HF={initial_hf:.2f}<extra></extra>",
        ))

        fig_yield.update_layout(
            xaxis_title="Boosted Liquidation LTV (λ̃_t)",
            yaxis_title="Health Factor",
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_yield, width='stretch')

    with col2:
        st.subheader("Days to Liquidation")

        # Cap days values for display (infinite -> 365)
        dtl_display = np.clip(dtl_grid, 0, 365)
        dtl_display[np.isinf(dtl_grid)] = 365

        # Identify cells with positive yield (SAFE cells)
        safe_mask = yield_grid > 0

        # Create custom hover text: "SAFE" for positive yield, days otherwise
        hover_text = []
        for i, hf_val in enumerate(hf_values):
            row_text = []
            for j, ltv_val in enumerate(liq_ltv_values):
                if safe_mask[i, j]:
                    row_text.append(f"Liq LTV: {ltv_val*100:.1f}%<br>HF: {hf_val:.2f}<br><b>SAFE</b> (yield > 0)")
                else:
                    row_text.append(f"Liq LTV: {ltv_val*100:.1f}%<br>HF: {hf_val:.2f}<br>Days: {dtl_display[i, j]:.0f}")
            hover_text.append(row_text)

        # Create masked arrays for separate plotting
        # For SAFE cells (positive yield): use purple
        # For risk cells (negative yield): use RdYlGn colorscale

        # Create display values: set SAFE cells to NaN for the risk heatmap
        dtl_risk = dtl_display.copy().astype(float)
        dtl_risk[safe_mask] = np.nan

        # Create SAFE values: 1 for SAFE, NaN otherwise
        dtl_safe = np.where(safe_mask, 1.0, np.nan)

        fig_dtl = go.Figure()

        # Layer 1: Risk heatmap (red-yellow-green for cells with negative yield)
        fig_dtl.add_trace(go.Heatmap(
            z=dtl_risk,
            x=[f"{v*100:.1f}%" for v in liq_ltv_values],
            y=[f"{v:.2f}" for v in hf_values],
            colorscale='RdYlGn',
            zmin=0,
            zmax=365,
            colorbar=dict(title="Days", x=1.02),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=True,
        ))

        # Layer 2: SAFE heatmap (purple for cells with positive yield)
        fig_dtl.add_trace(go.Heatmap(
            z=dtl_safe,
            x=[f"{v*100:.1f}%" for v in liq_ltv_values],
            y=[f"{v:.2f}" for v in hf_values],
            colorscale=[[0, '#7B2D8E'], [1, '#7B2D8E']],  # Solid purple
            zmin=0,
            zmax=1,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=False,
        ))

        # Add star marker for current position
        fig_dtl.add_trace(go.Scatter(
            x=[f"{current_liq_ltv*100:.1f}%"],
            y=[f"{initial_hf:.2f}"],
            mode='markers',
            marker=dict(size=18, color='#F18F01', symbol='star', line=dict(width=2, color='black')),
            name=f'Current ({leverage:.0f}x)',
            hovertemplate=f"Current Position<br>L={leverage:.0f}x, HF={initial_hf:.2f}<extra></extra>",
        ))

        fig_dtl.update_layout(
            xaxis_title="Boosted Liquidation LTV (λ̃_t)",
            yaxis_title="Health Factor",
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_dtl, width='stretch')

    # Explanation
    with st.expander("📖 How to read these matrices"):
        st.markdown("""
        **Axes:**
        - **Y-axis (Health Factor)**: Higher HF = safer position, lower effective leverage
        - **X-axis (Liq LTV)**: Higher = more Twyne boost, requires more CLP collateral

        **Yield Heatmap:**
        - 🟢 Green = Positive yield (profitable)
        - 🔴 Red = Negative yield (losing money)
        - The yield formula: `Y = (r_stake - r_borrow·λ_t - IR(u)·Ψ) / (1 - λ_t)`

        **Days to Liquidation:**
        - Shows how many days until HF decays to 1.0 (if r_net < 0)
        - 🟣 Purple = **SAFE** — yield is positive, no liquidation risk
        - 🟢 Green = Many days until liquidation
        - 🔴 Red = Fewer days to react before liquidation
        """)

# =============================================================================
# Tab 2: Yield Calculator
# =============================================================================

with tab2:
    st.header("Yield Calculator")

    # Current parameters display
    col1, col2, col3 = st.columns(3)

    # Calculate derived values
    lambda_t = ltv_from_leverage(leverage)
    liq_ltv_t = lambda_t * initial_hf
    psi_val = psi(liq_ltv_t, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])

    with col1:
        st.metric("Operating LTV (λ_t)", f"{lambda_t*100:.1f}%")
        st.metric("Boosted Liq LTV (λ̃_t)", f"{liq_ltv_t*100:.1f}%")

    with col2:
        st.metric("CLP Cost Factor (Ψ)", f"{psi_val:.3f}")
        st.metric("Interest Rate IR(u)", f"{ir_u*100:.2f}%")

    with col3:
        credit_cost = psi_val * ir_u
        r_net_val = r_stake - r_borrow - credit_cost
        st.metric("Credit Delegation Cost", f"{credit_cost*100:.2f}%")
        st.metric("Net Rate (r_net)", f"{r_net_val*100:.2f}%",
                  delta="Safe" if r_net_val >= 0 else "Decaying")

    st.divider()

    # Main yield calculation
    yield_val = looped_yield_from_leverage(r_stake, r_borrow, leverage, ir_u, psi_val)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "📊 Net Yield (Annualized)",
            f"{yield_val*100:.2f}%",
            delta=f"{leverage}x leverage"
        )

        if r_net_val >= 0:
            st.success("✅ Position is profitable - HF will improve over time")
        else:
            dtl = days_to_liquidation(initial_hf, r_stake, r_borrow, ir_u, psi_val)
            st.warning(f"⚠️ Negative net rate - ~{dtl:.0f} days to liquidation")

    with col2:
        # Yield vs Leverage curve
        L_range = np.linspace(2, 50, 100)
        yields = []
        for L in L_range:
            lt = ltv_from_leverage(L)
            llt = lt * initial_hf
            p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
            y = looped_yield_from_leverage(r_stake, r_borrow, L, ir_u, p)
            yields.append(y * 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=L_range, y=yields,
            mode='lines', name='Net Yield',
            line=dict(color='#7B2CBF', width=3),
        ))

        # Mark current leverage
        fig.add_trace(go.Scatter(
            x=[leverage], y=[yield_val * 100],
            mode='markers', name=f'Current ({leverage}x)',
            marker=dict(size=15, color='#F18F01', symbol='star'),
        ))

        # Add traditional Aave limit
        fig.add_vline(x=5, line_dash="dash", line_color="gray",
                      annotation_text="Aave max (~5x)")

        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)

        fig.update_layout(
            title="Yield vs Leverage",
            xaxis_title="Leverage (x)",
            yaxis_title="Net Yield (%)",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig, width='stretch')

    # Breakdown table
    st.subheader("Yield Breakdown")

    breakdown_data = {
        "Component": [
            "Collateral Yield (L × r_stake)",
            "Borrow Cost ((L-1) × r_borrow)",
            "Credit Delegation Cost (L × Ψ × IR)",
            "**Net Yield**"
        ],
        "Rate": [
            f"{leverage * r_stake * 100:.2f}%",
            f"-{(leverage - 1) * r_borrow * 100:.2f}%",
            f"-{leverage * credit_cost * 100:.2f}%",
            f"**{yield_val * 100:.2f}%**"
        ],
    }
    st.table(pd.DataFrame(breakdown_data))

# =============================================================================
# Tab 3: Historical Analysis
# =============================================================================

with tab3:
    st.header("Historical Analysis")

    # Summary stats
    stats = get_summary_stats(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Date Range", f"{stats['date_range']['n_days']} days")
    col2.metric("Mean Spread", f"{stats['spread']['mean']*100:.2f}%")
    col3.metric("Days Positive", f"{stats['spread']['pct_positive']:.1f}%")
    col4.metric("Min Spread", f"{stats['spread']['min']*100:.2f}%")

    # Historical rates plot
    st.subheader("Historical Rates")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Rates", "Spread"))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['r_stake'] * 100,
        name='stETH Yield', line=dict(color='#28A745'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['r_borrow'] * 100,
        name='ETH Borrow', line=dict(color='#DC3545'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['spread_raw'] * 100,
        name='Raw Spread', fill='tozeroy',
        line=dict(color='#2E86AB'),
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="black")

    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="APR (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=2, col=1)

    st.plotly_chart(fig, width='stretch')

    # Yield at different leverage levels
    st.subheader("Historical Yield by Leverage")

    leverage_levels = st.multiselect(
        "Select leverage levels to compare:",
        options=[5, 10, 15, 20, 30, 50],
        default=[5, 10, 20]
    )

    if leverage_levels:
        fig = go.Figure()

        colors = px.colors.qualitative.Set2

        for i, L in enumerate(sorted(leverage_levels)):
            yields = []
            for _, row in df.iterrows():
                lt = ltv_from_leverage(L)
                llt = lt * 1.05  # Assume HF=1.05
                p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
                y = looped_yield_from_leverage(row['r_stake'], row['r_borrow'], L, ir_u, p)
                yields.append(y * 100)

            fig.add_trace(go.Scatter(
                x=df.index, y=yields,
                name=f'{L}x',
                line=dict(color=colors[i % len(colors)]),
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Net Yield Over Time (at different leverage levels)",
            xaxis_title="Date",
            yaxis_title="Net Yield (%)",
            height=500,
        )
        st.plotly_chart(fig, width='stretch')

    # Spread distribution
    st.subheader("Spread Distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['spread_raw'] * 100,
        nbinsx=50,
        marker_color='#2E86AB',
        opacity=0.7,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    fig.add_vline(x=df['spread_raw'].mean() * 100, line_dash="solid",
                  line_color="green", annotation_text="Mean")

    fig.update_layout(
        title="Distribution of Daily Raw Spread",
        xaxis_title="Spread (%)",
        yaxis_title="Frequency",
        height=400,
    )
    st.plotly_chart(fig, width='stretch')

# =============================================================================
# Tab 4: Backtest
# =============================================================================

with tab4:
    st.header("Historical Backtest")

    st.markdown("""
    Simulate entering a leveraged position at the start of the historical data
    and track performance over time.
    """)

    # Backtest parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        bt_leverage_levels = st.multiselect(
            "Leverage levels to backtest:",
            options=[5, 10, 15, 20, 30, 40, 50],
            default=[5, 10, 20, 50]
        )

    with col2:
        bt_initial_hf = st.slider(
            "Initial HF for backtest",
            min_value=1.01, max_value=1.20, value=1.05, step=0.01
        )

    with col3:
        bt_utilization = st.slider(
            "Utilization for backtest (%)",
            min_value=0, max_value=100, value=95, step=5
        ) / 100

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtests..."):
            results = run_multi_leverage_backtest(
                df,
                leverage_levels=bt_leverage_levels,
                initial_hf=bt_initial_hf,
                utilization=bt_utilization,
                **params
            )

        # Results table
        st.subheader("Results Summary")

        summary_data = []
        for L in sorted(results.keys()):
            r = results[L]
            summary_data.append({
                "Leverage": f"{L}x",
                "Total Return": f"{r.metrics['total_return']*100:.1f}%",
                "Annualized": f"{r.metrics['annualized_return']*100:.1f}%",
                "Max Drawdown": f"{r.metrics['max_drawdown']*100:.1f}%",
                "Sharpe": f"{r.metrics['sharpe_ratio']:.2f}" if not np.isnan(r.metrics['sharpe_ratio']) else "N/A",
                "Min HF": f"{r.metrics['min_hf']:.3f}",
                "Liquidated": "❌ Yes" if r.metrics['was_liquidated'] else "✅ No",
            })

        st.dataframe(pd.DataFrame(summary_data), width='stretch')

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Capital Evolution")
            fig = go.Figure()
            for L in sorted(results.keys()):
                r = results[L]
                fig.add_trace(go.Scatter(
                    x=r.dates, y=r.capital,
                    name=f'{L}x',
                    mode='lines',
                ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Capital (starting at 1.0)",
                height=400,
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("Health Factor Evolution")
            fig = go.Figure()
            for L in sorted(results.keys()):
                r = results[L]
                fig.add_trace(go.Scatter(
                    x=r.dates, y=r.hf,
                    name=f'{L}x',
                    mode='lines',
                ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                          annotation_text="Liquidation threshold")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Health Factor",
                height=400,
            )
            st.plotly_chart(fig, width='stretch')

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("""
**Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results.
Always understand the risks before participating in DeFi.

Built with the Twyne stETH-ETH Looping Economics module.
""")
