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

@st.cache_data(ttl=60)  # Refresh every 60 seconds
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

# Hardcoded: Ethereum chain, Aave market, wstETH CLP asset
selected_chain = "Ethereum"
lending_market = "Aave"
clp_asset = "wstETH"

# Default liquidation thresholds
default_underlying_lltv = 0.93  # Aave e-mode (same for C and CLP)
default_twyne_lltv_max = 0.97  # Twyne max boosted LLTV

# =============================================================================
# Rate Source (logic only - UI moved to Yield Analysis tab)
# =============================================================================

# Initialize session state for rates
if 'live_rates' not in st.session_state:
    st.session_state.live_rates = None
if 'live_rates_chain' not in st.session_state:
    st.session_state.live_rates_chain = selected_chain
if 'rate_source' not in st.session_state:
    st.session_state.rate_source = "Current"
if 'custom_r_steth_stake' not in st.session_state:
    st.session_state.custom_r_steth_stake = 3.5
if 'custom_r_steth_lend' not in st.session_state:
    st.session_state.custom_r_steth_lend = 0.1
if 'custom_r_borrow' not in st.session_state:
    st.session_state.custom_r_borrow = 2.5

# Auto-fetch live rates if not cached
if st.session_state.live_rates is None or st.session_state.live_rates_chain != selected_chain:
    st.session_state.live_rates = fetch_live_rates(selected_chain)
    st.session_state.live_rates_chain = selected_chain

# Determine rates based on source
if st.session_state.rate_source == "Current" and st.session_state.live_rates is not None:
    live = st.session_state.live_rates
    r_steth_stake = live.steth_staking_rate
    r_steth_lend = live.steth_supply_rate
    r_stake = live.r_stake
    r_borrow = live.eth_borrow_rate
else:
    r_steth_stake = st.session_state.custom_r_steth_stake / 100
    r_steth_lend = st.session_state.custom_r_steth_lend / 100
    r_stake = r_steth_stake + r_steth_lend
    r_borrow = st.session_state.custom_r_borrow / 100

# =============================================================================
# LTVs & Safety Buffer
# =============================================================================
with st.sidebar.container(border=True):
    st.caption("**LTVs & Safety Buffer**")
    underlying_lltv = st.slider(
        "Underlying LLTV (λ̃_e) (%)",
        min_value=70, max_value=96, value=93, step=1,
        help="Aave e-mode liquidation threshold (same for C and CLP)",
        key="underlying_lltv"
    ) / 100

    twyne_lltv_max = st.slider(
        "Twyne Max LLTV (λ̃_t^max) (%)",
        min_value=90, max_value=99, value=97, step=1,
        help="Twyne's maximum boosted liquidation LTV",
        key="twyne_liq_ltv_max"
    ) / 100

    beta_safe = st.slider(
        "Safety Buffer (β_safe)",
        min_value=0.85, max_value=1.00, value=1.00, step=0.01,
        help="Safety buffer from whitepaper Eq. 7",
        key="twyne_beta_safe"
    )

st.sidebar.divider()

# =============================================================================
# CLP Utilization
# =============================================================================
with st.sidebar.container(border=True):
    st.caption("**CLP Utilization**")
    utilization = st.slider(
        "Utilization (u) (%)",
        min_value=0, max_value=100, value=95, step=5,
        help="Credit LP pool utilization - affects IR(u)",
        key="twyne_utilization"
    ) / 100

st.sidebar.divider()

# =============================================================================
# Interest Rate Model
# =============================================================================
with st.sidebar.container(border=True):
    st.caption("**Interest Rate Model**")
    ir_I_min = st.slider("I_min (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="ir_I_min") / 100
    ir_I_max = st.slider("I_max (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="ir_I_max") / 100
    ir_I_0 = st.slider("I_0 (%)", min_value=0.0, max_value=float(st.session_state.get("ir_I_max", 0.5)), value=min(0.8, float(st.session_state.get("ir_I_max", 0.5))), step=0.1, key="ir_I_0") / 100
    ir_u_0 = st.slider("u_0 (%)", min_value=50, max_value=99, value=90, step=1, key="ir_u_0") / 100
    ir_gamma = st.slider("γ (curvature)", min_value=1, max_value=64, value=32, step=1, key="ir_gamma")

# Get params from session state
underlying_lltv = st.session_state.get("underlying_lltv", 93) / 100
liq_ltv_c = underlying_lltv  # Same for collateral
liq_ltv_clp = underlying_lltv  # Same for CLP
liq_ltv_max = st.session_state.get("twyne_liq_ltv_max", 97) / 100
utilization = st.session_state.get("twyne_utilization", 95) / 100
beta_safe = st.session_state.get("twyne_beta_safe", 1.00)
ir_I_min = st.session_state.get("ir_I_min", 0.0) / 100
ir_I_0 = st.session_state.get("ir_I_0", 0.8) / 100
ir_u_0 = st.session_state.get("ir_u_0", 90) / 100
ir_I_max = st.session_state.get("ir_I_max", 0.5) / 100
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

# Calculate leverage bounds (used across multiple tabs)
L_aave = 1.0 / (1.0 - liq_ltv_c)  # ~14x for 93% e-mode
L_max_global = 1.0 / (1.0 - liq_ltv_max)  # 50x for 98% max

# =============================================================================
# Color Scheme Helper (Aave: yellow/orange, Twyne: blue/purple)
# =============================================================================

AAVE_COLORS = {
    2: '#fff176',   # Light yellow
    3: '#fff176',
    4: '#ffee58',   # Yellow
    5: '#ffee58',
    6: '#fdd835',   # Golden yellow
    7: '#fdd835',
    8: '#ffb300',   # Amber
    9: '#ffb300',
    10: '#fb8c00',  # Orange
    11: '#fb8c00',
    12: '#f57c00',  # Dark orange
    13: '#f57c00',
    14: '#ef6c00',  # Deep orange
}

TWYNE_COLORS = {
    15: '#42a5f5',  # Light blue
    16: '#42a5f5',
    17: '#42a5f5',
    18: '#2196f3',  # Blue
    19: '#2196f3',
    20: '#1e88e5',  # Medium blue
    25: '#1565c0',  # Darker blue
    30: '#5e35b1',  # Deep purple
    35: '#7b1fa2',  # Purple
    40: '#8e24aa',  # Magenta purple
    45: '#6a1b9a',  # Dark purple
    50: '#4a148c',  # Deepest purple
}

def get_leverage_style(L):
    """Returns (name, color, width) for a leverage level."""
    aave_max = int(L_aave)
    twyne_max = int(L_max_global)

    if L == aave_max:
        # Aave max leverage
        color = AAVE_COLORS.get(L, '#606060')
        return f'Aave (max) {L}x', color, 2.0
    elif L < aave_max:
        # Aave - below max
        color = AAVE_COLORS.get(L, '#606060')
        return f'Aave {L}x', color, 2.0
    elif L == twyne_max:
        # Twyne max leverage
        closest = min(TWYNE_COLORS.keys(), key=lambda x: abs(x - L))
        color = TWYNE_COLORS[closest]
        return f'Twyne (max) {L}x', color, 2.5
    else:
        # Twyne - blue to purple (find closest defined)
        closest = min(TWYNE_COLORS.keys(), key=lambda x: abs(x - L))
        color = TWYNE_COLORS[closest]
        return f'Twyne {L}x', color, 2.5

# =============================================================================
# Main Content - Tabs
# =============================================================================

st.title("🔄 Twyne stETH-ETH Looping Economics")

tab1, tab2 = st.tabs([
    "📈 Backtest & History",
    "📊 Yield Analysis"
])

# =============================================================================
# Tab 2: Yield Analysis (Detailed Sheet)
# =============================================================================

with tab2:
    st.header("Yield Analysis")

    # -------------------------------------------------------------------------
    # Rate Source Controls
    # -------------------------------------------------------------------------
    rate_col1, rate_col2 = st.columns([1, 2])

    with rate_col1:
        rate_source = st.radio(
            "Rate Source",
            ["Current", "Custom"],
            index=0 if st.session_state.rate_source == "Current" else 1,
            horizontal=True,
            key="rate_source_radio"
        )
        st.session_state.rate_source = rate_source

        if rate_source == "Current":
            if st.button("🔄 Refresh Rates"):
                st.session_state.live_rates = fetch_live_rates(selected_chain)
                st.rerun()

    with rate_col2:
        if rate_source == "Custom":
            custom_cols = st.columns(3)
            with custom_cols[0]:
                st.session_state.custom_r_steth_stake = st.number_input(
                    "stETH Staking %", min_value=0.0, max_value=10.0,
                    value=st.session_state.custom_r_steth_stake, step=0.1, format="%.2f"
                )
            with custom_cols[1]:
                st.session_state.custom_r_steth_lend = st.number_input(
                    "stETH Lending %", min_value=0.0, max_value=5.0,
                    value=st.session_state.custom_r_steth_lend, step=0.05, format="%.2f"
                )
            with custom_cols[2]:
                st.session_state.custom_r_borrow = st.number_input(
                    "ETH Borrow %", min_value=0.0, max_value=25.0,
                    value=st.session_state.custom_r_borrow, step=0.1, format="%.2f"
                )
            # Update rates from custom values
            r_steth_stake = st.session_state.custom_r_steth_stake / 100
            r_steth_lend = st.session_state.custom_r_steth_lend / 100
            r_stake = r_steth_stake + r_steth_lend
            r_borrow = st.session_state.custom_r_borrow / 100

    # Rate metrics display
    spread = r_stake - r_borrow
    metric_cols = st.columns(5)
    metric_cols[0].metric("stETH Staking (Lido)", f"{r_steth_stake*100:.2f}%")
    metric_cols[1].metric("stETH Lending (Aave)", f"{r_steth_lend*100:.2f}%")
    metric_cols[2].metric("Total stETH Yield", f"{r_stake*100:.2f}%", help="Staking + Lending")
    metric_cols[3].metric("ETH Borrow (Aave)", f"{r_borrow*100:.2f}%")
    metric_cols[4].metric("Net Spread", f"{spread*100:.2f}%",
                          delta="Positive" if spread > 0 else "Negative")

    st.divider()

    # -------------------------------------------------------------------------
    # Yield & Days to Liquidation Heatmaps
    # -------------------------------------------------------------------------

    # HF range inputs will be shown below the charts
    # Use session state to preserve values
    if 'hf_min' not in st.session_state:
        st.session_state.hf_min = 1.01
    if 'hf_max' not in st.session_state:
        st.session_state.hf_max = 1.10

    hf_min = st.session_state.hf_min
    hf_max = st.session_state.hf_max

    # LTV values: start at Aave max (only show Twyne-boosted range)
    ltv_min = liq_ltv_c  # Start at Aave max (e.g., 93%)
    ltv_max = liq_ltv_max  # Twyne max (e.g., 98%)

    n_hf = 10
    hf_values = np.linspace(hf_max, hf_min, n_hf)  # High to low

    # LTV values from Aave max to Twyne max: [aave, twyne1, twyne2, twyne3, twyne_max]
    liq_ltv_values = np.array([
        liq_ltv_c,         # e.g., 93% - Aave max leverage
        liq_ltv_c + (ltv_max - liq_ltv_c) * 0.25,  # 25% between
        (liq_ltv_c + ltv_max) / 2,  # midpoint
        liq_ltv_c + (ltv_max - liq_ltv_c) * 0.75,  # 75% between
        ltv_max,           # e.g., 98% - Twyne max
    ])

    # Compute grids
    yield_grid = compute_yield_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                     utilization=utilization, **params)
    dtl_grid = compute_days_to_liq_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                         utilization=utilization, **params)

    # Toggle for absolute vs delta yields + HF inputs on the right
    yield_mode_col, _, hf_col1, hf_col2 = st.columns([2, 1, 1, 1])
    with yield_mode_col:
        yield_mode = st.segmented_control(
            "Yield Display",
            ["Absolute", "Delta (vs Aave)"],
            default="Absolute",
            label_visibility="collapsed",
            key="yield_display_mode"
        )
    with hf_col1:
        new_hf_min = st.number_input("HF Min", min_value=1.01, max_value=1.5, value=st.session_state.hf_min, step=0.01, format="%.2f", key="hf_min_input")
        if new_hf_min != st.session_state.hf_min:
            st.session_state.hf_min = new_hf_min
            st.rerun()
    with hf_col2:
        new_hf_max = st.number_input("HF Max", min_value=1.02, max_value=2.0, value=st.session_state.hf_max, step=0.01, format="%.2f", key="hf_max_input")
        if new_hf_max != st.session_state.hf_max:
            st.session_state.hf_max = new_hf_max
            st.rerun()

    # Always show both heatmaps side by side
    col1, col2 = st.columns(2)

    # Aave threshold in LTV terms (no Twyne boost needed below this)
    aave_lltv = liq_ltv_c  # underlying LLTV (e.g., 93%)

    # Use numeric x values (LTV as percentage number) for proper shape positioning
    x_values = liq_ltv_values * 100  # e.g., [92, 93.5, 95, 96.5, 98]
    y_values = hf_values  # e.g., [1.20, 1.18, ..., 1.00]

    with col1:
        # Compute display values based on mode
        yield_pct = yield_grid * 100

        if yield_mode == "Delta (vs Aave)":
            # Subtract the Aave baseline (first column) from each row
            aave_baseline = yield_pct[:, 0:1]  # First column (Aave max LTV)
            display_values = yield_pct - aave_baseline
            title = "**Extra Yield vs Aave**"
            caption = "Additional yield from Twyne vs Aave max leverage. Shows the benefit of boosted liquidation LTV."
            colorbar_title = "Extra %"
            text_annotations = [[f"+{val:.1f}%" if val >= 0 else f"{val:.1f}%" for val in row] for row in display_values]
        else:
            display_values = yield_pct
            title = "**Net Yield Heatmap**"
            caption = "Theoretical yield at different Health Factors and Liquidation LTVs. Green = profitable, Red = losing money."
            colorbar_title = "Yield %"
            text_annotations = [[f"{val:.1f}%" for val in row] for row in display_values]

        st.markdown(title)
        st.caption(caption)

        # Custom colorscale: red -> yellow -> light green -> green -> teal -> blue
        custom_colorscale = [
            [0.0, '#d73027'],    # Dark red (most negative)
            [0.3, '#fc8d59'],    # Orange-red
            [0.45, '#fee08b'],   # Yellow (near zero negative)
            [0.5, '#ffffbf'],    # Light yellow (zero)
            [0.55, '#d9ef8b'],   # Light green (near zero positive)
            [0.7, '#91cf60'],    # Green
            [0.85, '#1a9850'],   # Dark green
            [1.0, '#006837'],    # Deep green (most positive)
        ]

        fig_yield = go.Figure(data=go.Heatmap(
            z=display_values,
            x=x_values,
            y=y_values,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=custom_colorscale,
            zmid=0,  # Center at 0 so negative=red, positive=green
            colorbar=dict(title=colorbar_title),
            hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br>Value: %{z:.2f}%<extra></extra>",
        ))

        # Add vertical line for Aave threshold at left edge
        aave_x = aave_lltv * 100  # e.g., 93
        fig_yield.add_vline(
            x=aave_x + 0.3,  # Slightly to the right so it doesn't overlap first column
            line=dict(color="blue", width=3, dash="dash"),
            annotation_text=f"Aave max ({aave_x:.0f}%)",
            annotation_position="top right",
            annotation=dict(font_color="blue", font_size=11),
        )

        fig_yield.update_layout(
            xaxis_title="Boosted Liquidation LTV (λ̃_t) %",
            yaxis_title="Health Factor",
            height=400,
            margin=dict(t=20),
            xaxis=dict(
                tickmode='array',
                tickvals=x_values,
                ticktext=[f"{v:.1f}" for v in x_values],
            ),
            yaxis=dict(
                range=[hf_min - 0.01, hf_max + 0.01],
                tickmode='array',
                tickvals=y_values,
                ticktext=[f"{v:.2f}" for v in y_values],
            ),
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with col2:
        st.markdown("**Days to Liquidation**")
        st.caption("Time until HF reaches 1.0 (liquidation). Purple = SAFE (yield covers costs). Lower = riskier.")

        # Cap days values for display (infinite -> 365)
        dtl_display = np.clip(dtl_grid, 0, 365)
        dtl_display[np.isinf(dtl_grid)] = 365

        # Identify cells with positive yield (SAFE cells)
        safe_mask = yield_grid > 0

        # Create text annotations: "Safe" for positive yield, days for negative
        dtl_text = []
        for i, row in enumerate(dtl_display):
            text_row = []
            for j, val in enumerate(row):
                if safe_mask[i, j]:
                    text_row.append("Safe")
                else:
                    text_row.append(f"{int(val)}d")
            dtl_text.append(text_row)

        # Create masked arrays for separate plotting
        dtl_risk = dtl_display.copy().astype(float)
        dtl_risk[safe_mask] = np.nan
        dtl_safe = np.where(safe_mask, 1.0, np.nan)

        fig_dtl = go.Figure()

        # Layer 1: Risk heatmap (red-yellow-green for cells with negative yield)
        fig_dtl.add_trace(go.Heatmap(
            z=dtl_risk,
            x=x_values,
            y=y_values,
            colorscale='RdYlGn',
            zmin=0,
            zmax=365,
            colorbar=dict(title="Days", x=1.02),
            hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br>Days: %{z:.0f}<extra></extra>",
            showscale=True,
        ))

        # Layer 2: SAFE heatmap (purple for cells with positive yield)
        fig_dtl.add_trace(go.Heatmap(
            z=dtl_safe,
            x=x_values,
            y=y_values,
            colorscale=[[0, '#7B2D8E'], [1, '#7B2D8E']],  # Solid purple
            zmin=0,
            zmax=1,
            hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br><b>SAFE</b><extra></extra>",
            showscale=False,
        ))

        # Add text annotations as a separate layer
        for i, hf in enumerate(y_values):
            for j, ltv in enumerate(x_values):
                fig_dtl.add_annotation(
                    x=ltv,
                    y=hf,
                    text=dtl_text[i][j],
                    showarrow=False,
                    font=dict(size=10, color="black" if not safe_mask[i, j] else "white"),
                )

        # Add vertical line for Aave threshold at left edge
        fig_dtl.add_vline(
            x=aave_x + 0.3,  # Slightly to the right so it doesn't overlap first column
            line=dict(color="blue", width=3, dash="dash"),
            annotation_text=f"Aave max ({aave_x:.0f}%)",
            annotation_position="top right",
            annotation=dict(font_color="blue", font_size=11),
        )

        fig_dtl.update_layout(
            xaxis_title="Boosted Liquidation LTV (λ̃_t) %",
            yaxis_title="Health Factor",
            height=400,
            margin=dict(t=20),
            showlegend=False,
            xaxis=dict(
                tickmode='array',
                tickvals=x_values,
                ticktext=[f"{v:.1f}" for v in x_values],
            ),
            yaxis=dict(
                range=[hf_min - 0.01, hf_max + 0.01],
                tickmode='array',
                tickvals=y_values,
                ticktext=[f"{v:.2f}" for v in y_values],
            ),
        )
        st.plotly_chart(fig_dtl, use_container_width=True)

    # =========================================================================
    # Twyne Credit Efficiency
    # =========================================================================
    # Calculate bounds for efficiency calculation
    L_aave = 1.0 / (1.0 - liq_ltv_c)  # ~14x for 93% e-mode
    aave_max_int = int(L_aave)
    twyne_max_int = int(L_max_global)
    ref_hf = 1.05

    # Helper to calculate yield for a leverage level
    def calc_yield_for_leverage(L, r_stk, r_brw):
        """Calculate net yield for leverage L at given rates."""
        lt = ltv_from_leverage(L)
        llt = lt * ref_hf
        is_twyne = L > L_aave
        if is_twyne:
            p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
            credit_cost = L * p * ir_u
        else:
            p = 0
            credit_cost = 0
        return (L * r_stk - (L - 1) * r_brw - credit_cost) * 100
    st.markdown("**Twyne Credit Efficiency**")
    st.caption("How yield components change as you increase leverage from Aave max to Twyne max.")

    aave_L = aave_max_int
    aave_max_yield = calc_yield_for_leverage(aave_L, r_stake, r_borrow)

    # Generate data for all leverage levels from 1x to Twyne max
    leverage_range = np.arange(1, twyne_max_int + 1)
    all_data = {}

    for L in leverage_range:
        # Calculate yield (Aave formula for all, no credit cost below Aave max)
        L_yield = calc_yield_for_leverage(L, r_stake, r_borrow)

        # Credit cost = L × Ψ × IR(u)
        # - Scales with L (total collateral in the leveraged position)
        # - Ψ is small because Twyne only needs a little credit to boost the liquidation LTV
        if L > aave_L:
            extra_leverage = L - aave_L
            lt = ltv_from_leverage(L)
            llt = lt * ref_hf
            p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
            credit_cost = L * p * ir_u * 100
        else:
            credit_cost = 0
            extra_leverage = 0

        extra_yield = L_yield - aave_max_yield
        gross_extra = extra_yield + credit_cost
        efficiency_ratio = gross_extra / credit_cost if credit_cost > 0 else float('inf')

        breakeven_spread = (credit_cost / 100) / extra_leverage if extra_leverage > 0 else 0

        all_data[L] = {
            'yield': L_yield,
            'credit_cost': credit_cost,
            'extra_yield': extra_yield,
            'gross_extra': gross_extra,
            'efficiency': efficiency_ratio,
            'breakeven': breakeven_spread,
        }

    # For waterfall examples, use twyne_data alias
    twyne_data = all_data

    # Create continuous chart with dual y-axis
    fig_eff = go.Figure()

    # Extract data for plotting
    leverages = list(leverage_range)
    all_yields = [all_data[L]['yield'] for L in leverages]
    credit_costs = [all_data[L]['credit_cost'] for L in leverages]

    # Split into Aave range (1 to 14) and Twyne range (14+)
    aave_leverages = [L for L in leverages if L <= aave_max_int]
    twyne_leverages = [L for L in leverages if L >= aave_max_int]
    aave_yields = [all_data[L]['yield'] for L in aave_leverages]
    twyne_yields = [all_data[L]['yield'] for L in twyne_leverages]
    twyne_credit_costs = [all_data[L]['credit_cost'] for L in twyne_leverages]
    # Gross yield = net yield + credit cost (what you'd earn without Twyne fees)
    twyne_gross_yields = [all_data[L]['yield'] + all_data[L]['credit_cost'] for L in twyne_leverages]

    # Aave yield line (1x to 14x) - no credit cost
    fig_eff.add_trace(go.Scatter(
        x=aave_leverages,
        y=aave_yields,
        mode='lines',
        name=f'Aave Yield (1-{aave_max_int}x)',
        line=dict(color='#ef6c00', width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 108, 0, 0.15)',
        hovertemplate='Aave %{x}x: %{y:.1f}%<extra></extra>',
    ))

    # Twyne gross yield (dotted line) - yield before credit cost
    fig_eff.add_trace(go.Scatter(
        x=twyne_leverages,
        y=twyne_gross_yields,
        mode='lines',
        name='Gross Yield (before credit cost)',
        line=dict(color='#7b1fa2', width=2, dash='dot'),
        hovertemplate='Gross %{x}x: %{y:.1f}%<extra></extra>',
    ))

    # Twyne net yield line (14x to 33x) - fill between gross and net to show credit cost
    fig_eff.add_trace(go.Scatter(
        x=twyne_leverages,
        y=twyne_yields,
        mode='lines',
        name=f'Twyne Net Yield ({aave_max_int}-{twyne_max_int}x)',
        line=dict(color='#7b1fa2', width=3),
        fill='tonexty',
        fillcolor='rgba(183, 28, 28, 0.3)',  # Red fill for credit cost
        hovertemplate='Twyne %{x}x: %{y:.1f}%<extra></extra>',
        legendgroup='twyne_net',
        showlegend=False,
    ))

    # Separate legend entry for Twyne Net Yield (without fill)
    fig_eff.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        name=f'Twyne Net Yield ({aave_max_int}-{twyne_max_int}x)',
        line=dict(color='#7b1fa2', width=3),
    ))

    # Separate legend entry for Credit Cost (red area)
    fig_eff.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Credit Cost',
        marker=dict(size=15, color='rgba(183, 28, 28, 0.5)', symbol='square'),
    ))

    # Fill under Twyne net yield
    fig_eff.add_trace(go.Scatter(
        x=twyne_leverages,
        y=twyne_yields,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tozeroy',
        fillcolor='rgba(123, 31, 162, 0.15)',
        showlegend=False,
        hoverinfo='skip',
    ))

    # Add vertical line at Aave max
    fig_eff.add_vline(
        x=aave_max_int,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text=f"Aave max ({aave_max_int}x)",
        annotation_position="top",
    )

    fig_eff.update_layout(
        title='Yield vs Leverage: Full Range',
        xaxis_title='Leverage (x)',
        yaxis_title='Yield / Cost (%)',
        height=380,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(l=50, r=150, t=40, b=50),
        hovermode='x unified',
    )

    st.plotly_chart(fig_eff, width='stretch')

    # =========================================================================
    # Waterfall Chart + Efficiency Chart (side by side)
    # =========================================================================
    # Titles row
    wf_title_col, eff_title_col = st.columns(2)
    with wf_title_col:
        st.markdown("**Yield Breakdown: Aave → Twyne**")
        st.caption("How Twyne leverage builds on Aave yield. Green = extra yield, Red = credit cost.")
    with eff_title_col:
        st.markdown("**Credit Efficiency vs Leverage**")
        st.caption("Extra yield per credit cost at each leverage. Higher = better value.")

    # Charts row
    wf_col, eff_col = st.columns(2)

    # Use Twyne max for main waterfall
    twyne_L = twyne_max_int
    d = all_data[twyne_L]

    with wf_col:
        # Create waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            orientation='v',
            x=[f'Aave {aave_L}x', 'Extra Yield', 'Credit Cost', f'Twyne {twyne_L}x'],
            y=[aave_max_yield, d['gross_extra'], -d['credit_cost'], 0],
            measure=['absolute', 'relative', 'relative', 'total'],
            connector={'line': {'color': 'rgba(0,0,0,0.1)'}},
            decreasing={'marker': {'color': '#c62828'}},  # Red for cost
            increasing={'marker': {'color': '#4CAF50'}},  # Green for gains
            totals={'marker': {'color': '#7b1fa2'}},      # Purple for total
            text=[f'{aave_max_yield:.1f}%', f'+{d["gross_extra"]:.1f}%',
                  f'-{d["credit_cost"]:.1f}%', f'{d["yield"]:.1f}%'],
            textposition='outside',
            textfont=dict(size=14),
        ))

        max_y = aave_max_yield + d['gross_extra']
        fig_waterfall.update_layout(
            height=350,
            yaxis_title='Annualized Yield (%)',
            showlegend=False,
            yaxis=dict(range=[0, max_y * 1.2]),
            margin=dict(l=50, r=30, t=20, b=50),
        )

        st.plotly_chart(fig_waterfall, use_container_width=True)

    with eff_col:
        # Calculate efficiency for all Twyne leverage levels
        twyne_only_leverages = [L for L in leverage_range if L > aave_max_int]
        efficiencies = [all_data[L]['efficiency'] for L in twyne_only_leverages]

        fig_eff_curve = go.Figure()

        # Efficiency line
        fig_eff_curve.add_trace(go.Scatter(
            x=twyne_only_leverages,
            y=efficiencies,
            mode='lines+markers',
            name='Efficiency',
            line=dict(color='#7b1fa2', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(123, 31, 162, 0.15)',
            hovertemplate='%{x}x: %{y:.1f}x efficiency<extra></extra>',
        ))

        # Add reference line at 1x efficiency (break-even)
        fig_eff_curve.add_hline(
            y=1,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            annotation_text="1x",
            annotation_position="right",
        )

        fig_eff_curve.update_layout(
            height=350,
            xaxis_title="Leverage (x)",
            yaxis_title="Efficiency",
            margin=dict(l=50, r=30, t=20, b=50),
        )

        st.plotly_chart(fig_eff_curve, use_container_width=True)

    # -------------------------------------------------------------------------
    # Spread Sensitivity & Buffer Charts
    # -------------------------------------------------------------------------

    current_spread = r_stake - r_borrow

    # Fixed spread range: ±3%
    spread_min = -0.03
    spread_max = 0.03
    spread_range = np.linspace(spread_min, spread_max, 100)

    # Calculate yields across spread range for each leverage level (include 1x for reference)
    leverage_to_plot = [1, 2, aave_max_int, 20, twyne_max_int]
    colors_sensitivity = {
        1: '#4caf50',  # Green for 1x holding
        2: '#fff176',
        aave_max_int: '#ef6c00',
        20: '#1e88e5',
        twyne_max_int: '#7b1fa2'
    }
    labels_sensitivity = {
        1: 'Hold stETH (1x)',
        2: 'Aave 2x',
        aave_max_int: f'Aave (max) {aave_max_int}x',
        20: 'Twyne 20x',
        twyne_max_int: f'Twyne (max) {twyne_max_int}x'
    }

    fig_sensitivity = go.Figure()

    for L in leverage_to_plot:
        yields_at_spread = []
        lt = ltv_from_leverage(L)
        llt = lt * ref_hf
        is_twyne = L > L_aave

        if is_twyne:
            p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
            credit_cost = L * p * ir_u
        else:
            credit_cost = 0

        for s in spread_range:
            # At spread s: r_stake = r_borrow + s (assuming r_borrow stays constant)
            # yield = L * (r_borrow + s) - (L-1) * r_borrow - credit_cost
            #       = L * r_borrow + L * s - L * r_borrow + r_borrow - credit_cost
            #       = L * s + r_borrow - credit_cost
            # Simplified: yield ≈ L * s - credit_cost (ignoring small r_borrow term for Aave)
            # More accurate: use actual formula
            r_stk = r_borrow + s
            y = (L * r_stk - (L - 1) * r_borrow - credit_cost) * 100
            yields_at_spread.append(y)

        fig_sensitivity.add_trace(go.Scatter(
            x=spread_range * 100,  # Convert to percentage
            y=yields_at_spread,
            mode='lines',
            name=labels_sensitivity[L],
            line=dict(color=colors_sensitivity[L], width=3),
            hovertemplate=f"{labels_sensitivity[L]}<br>Spread: %{{x:.2f}}%<br>Yield: %{{y:.1f}}%<extra></extra>",
        ))

    # Add current spread vertical line
    fig_sensitivity.add_vline(
        x=current_spread * 100,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"Current ({current_spread*100:.2f}%)",
        annotation_position="top",
    )

    # Add zero yield horizontal line
    fig_sensitivity.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)

    # Zero spread line
    fig_sensitivity.add_vline(
        x=0,
        line_dash="solid",
        line_color="gray",
        line_width=1,
    )

    fig_sensitivity.update_layout(
        height=400,
        xaxis_title="Spread (stETH yield − ETH borrow) %",
        xaxis=dict(range=[-3, 3]),
        yaxis_title="Net Yield (%)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        hovermode='x unified',
        margin=dict(r=150),
    )

    # Calculate buffers for each leverage level (needed for buffer chart)
    # Show only one Aave level, labeled as "Aave"
    buffer_levels = [aave_max_int, 20, twyne_max_int]
    buffer_labels = ["Aave", "Twyne 20x", f"Twyne {twyne_max_int}x"]
    buffer_breakevens = []
    buffer_values = []
    buffer_colors = ["#ef6c00", "#1e88e5", "#7b1fa2"]

    for L in buffer_levels:
        # Calculate break-even spread
        if L > L_aave:
            lt = ltv_from_leverage(L)
            llt = lt * ref_hf
            p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
            credit_cost_buf = L * p * ir_u
            s_be = credit_cost_buf / (L - 1)
        else:
            s_be = 0  # Aave breaks even at spread = 0

        buffer = current_spread - s_be
        buffer_breakevens.append(s_be * 100)
        buffer_values.append(buffer * 100)

    fig_buffer = go.Figure()

    # Bars showing buffer (from breakeven to current spread)
    fig_buffer.add_trace(go.Bar(
        x=buffer_labels,
        y=buffer_values,
        marker_color=buffer_colors,
        text=[f"{v:.2f}%" for v in buffer_values],
        textposition="outside",
        hovertemplate='%{x}: Buffer = %{y:.2f}%<extra></extra>',
    ))

    # Add current spread reference line
    fig_buffer.add_hline(
        y=current_spread * 100,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text=f"Current ({current_spread*100:.2f}%)",
        annotation_position="top right",
        annotation_font_color="green",
    )

    fig_buffer.update_layout(
        height=400,
        yaxis_title="Buffer (%)",
        yaxis=dict(range=[0, max(buffer_values) * 1.3]),
        margin=dict(l=50, r=30, t=20, b=30),
    )

    # Side by side layout with titles above each
    col_title1, col_title2 = st.columns([2, 1])
    with col_title1:
        st.markdown("**Spread Sensitivity**")
        st.caption("How yield changes with spread.")
    with col_title2:
        st.markdown("**Spread Buffer**")
        st.caption("How much spread can drop before worse than holding stETH.")

    col_sens, col_buf = st.columns([2, 1])
    with col_sens:
        st.plotly_chart(fig_sensitivity, width='stretch')
    with col_buf:
        st.plotly_chart(fig_buffer, width='stretch')

    # =========================================================================
    # Rate Shock Analysis
    # =========================================================================
    st.markdown("**Rate Shock Analysis**")

    # Stress test scenarios
    scenarios = [
        {"name": "Current", "label": "Current", "borrow_mult": 1.0, "stake_mult": 1.0},
        {"name": "Mild", "label": "Mild\n(+50%)", "borrow_mult": 1.5, "stake_mult": 1.0},
        {"name": "Moderate", "label": "Moderate\n(+100%)", "borrow_mult": 2.0, "stake_mult": 1.0},
        {"name": "Severe", "label": "Severe\n(+200%)", "borrow_mult": 3.0, "stake_mult": 0.8},
    ]

    # Leverage levels (simplified - no HF variants)
    leverage_levels = [2, aave_max_int, 20, twyne_max_int]
    ref_hf_shock = 1.05  # Reference HF for calculations

    # Colors for each leverage level
    leverage_colors = {
        2: '#4caf50',  # Green
        aave_max_int: '#ff9800',  # Orange
        20: '#2196f3',  # Blue
        twyne_max_int: '#9c27b0',  # Purple
    }

    # Calculate data for each leverage level
    leverage_data = {}
    for L in leverage_levels:
        lt = ltv_from_leverage(L)
        llt = lt * ref_hf_shock
        psi_val = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])

        # Rate buffer
        break_even = (L * r_stake - L * psi_val * ir_u) / (L - 1)
        buffer_pct = (break_even / r_borrow - 1) * 100 if r_borrow > 0 else float('inf')

        label = f"{'Aave' if L <= aave_max_int else 'Twyne'} {L}x"
        yields = []
        dtl_values = []  # Days to liquidation
        for s in scenarios:
            shocked_borrow = r_borrow * s["borrow_mult"]
            shocked_stake = r_stake * s["stake_mult"]
            shocked_yield = (L * shocked_stake - (L - 1) * shocked_borrow - L * psi_val * ir_u) * 100
            yields.append(shocked_yield)

            # Calculate days to liquidation
            r_net_shocked = shocked_stake - shocked_borrow - psi_val * ir_u
            if r_net_shocked >= 0:
                dtl_values.append(None)  # Safe
            else:
                decay_factor = 1.0 + r_net_shocked / (1.0 + shocked_borrow)
                if decay_factor > 0 and decay_factor < 1:
                    dtl_days = -365 * np.log(ref_hf_shock) / np.log(decay_factor)
                    dtl_values.append(dtl_days)
                else:
                    dtl_values.append(None)

        leverage_data[L] = {
            'label': label,
            'buffer': buffer_pct,
            'yields': yields,
            'dtl': dtl_values,
            'color': leverage_colors.get(L, '#666'),
        }

    # Rate Buffer header and metrics
    st.markdown("**Rate Buffer** — Max borrow rate increase before losing money")
    metric_cols = st.columns(len(leverage_levels))
    for i, L in enumerate(leverage_levels):
        data = leverage_data[L]
        with metric_cols[i]:
            st.metric(
                data['label'],
                f"+{data['buffer']:.0f}%" if data['buffer'] < 1000 else "Safe",
                help=f"Borrow rate can increase by this much before {data['label']} yield turns negative"
            )

    st.markdown("**Impact of Borrow Rate Spikes**")
    st.caption("How yield and time-to-liquidation change as borrow rates increase. Steeper slope = higher sensitivity to rate shocks.")

    # Side by side charts
    yield_col, dtl_col = st.columns(2)

    with yield_col:
        st.markdown("**Net Yield**")
        fig = go.Figure()

        for L in leverage_levels:
            data = leverage_data[L]
            fig.add_trace(go.Scatter(
                x=[s["label"] for s in scenarios],
                y=data['yields'],
                mode='lines+markers+text',
                name=data['label'],
                line=dict(color=data['color'], width=3),
                marker=dict(size=8),
                text=[f"{y:.0f}%" for y in data['yields']],
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate=f"<b>{data['label']}</b><br>%{{x}}<br>Net Yield: %{{y:.1f}}%<extra></extra>",
            ))

        # Zero line (break-even)
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)

        # Calculate y-axis range
        all_yields = [y for L in leverage_levels for y in leverage_data[L]['yields']]
        min_y = min(all_yields)
        max_y = max(all_yields)
        padding = max(abs(min_y), abs(max_y)) * 0.25

        fig.update_layout(
            height=350,
            yaxis=dict(
                title="Net Yield (%)",
                range=[min_y - padding, max_y + padding],
                zeroline=False,
            ),
            xaxis=dict(title=""),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=1.02,
            ),
            margin=dict(l=50, r=100, t=10, b=40),
            hovermode='x unified',
        )

        st.plotly_chart(fig, use_container_width=True)

    with dtl_col:
        st.markdown("**Days to Liquidation**")
        fig_dtl = go.Figure()

        for L in leverage_levels:
            data = leverage_data[L]
            dtl_display = []
            dtl_text = []
            for dtl in data['dtl']:
                if dtl is None:
                    dtl_display.append(None)
                    dtl_text.append("Safe")
                else:
                    dtl_display.append(dtl)
                    dtl_text.append(f"{int(dtl)}d")

            fig_dtl.add_trace(go.Scatter(
                x=[s["label"] for s in scenarios],
                y=dtl_display,
                mode='lines+markers+text',
                name=data['label'],
                line=dict(color=data['color'], width=3),
                marker=dict(size=8),
                text=dtl_text,
                textposition='top center',
                textfont=dict(size=9),
                hovertemplate=f"<b>{data['label']}</b><br>%{{x}}<br>Days to Liq: %{{text}}<extra></extra>",
                connectgaps=False,
            ))

        # Calculate y-axis range for DTL
        all_dtl = [d for L in leverage_levels for d in leverage_data[L]['dtl'] if d is not None]
        if all_dtl:
            max_dtl = max(all_dtl)
            fig_dtl.update_layout(
                yaxis=dict(range=[0, max_dtl * 1.3]),
            )

        fig_dtl.update_layout(
            height=350,
            yaxis=dict(title="Days"),
            xaxis=dict(title=""),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=1.02,
            ),
            margin=dict(l=50, r=100, t=10, b=40),
            hovermode='x unified',
            showlegend=False,  # Hide legend on second chart (redundant)
        )

        st.plotly_chart(fig_dtl, use_container_width=True)

    st.caption("**Reading these charts:** Steeper yield decline = more sensitive to rate shocks. "
               "Days to Liquidation shows how long you have to react. 'Safe' = positive yield, no liquidation risk.")

# =============================================================================
# Tab 1: Backtest & History
# =============================================================================

with tab1:
    # =========================================================================
    # Backtest Section
    # =========================================================================
    st.header("Historical Backtest")

    # Backtest parameters in expander
    with st.expander("Backtest Settings", expanded=False):
        # Time range selector
        min_date = df.index.min().to_pydatetime()
        max_date = df.index.max().to_pydatetime()

        # Quick year selection buttons
        available_years = sorted(df.index.year.unique())
        year_cols = st.columns(len(available_years) + 1)

        # Initialize session state for selected year
        if 'selected_year' not in st.session_state:
            st.session_state.selected_year = None

        with year_cols[0]:
            if st.button("All", use_container_width=True):
                st.session_state.selected_year = None
                st.session_state.selected_relative = None

        for i, year in enumerate(available_years):
            with year_cols[i + 1]:
                if st.button(str(year), use_container_width=True):
                    st.session_state.selected_year = year
                    st.session_state.selected_relative = None

        # Relative time range buttons (second row)
        relative_options = [
            ("7d", 7),
            ("30d", 30),
            ("3mo", 90),
            ("6mo", 180),
            ("1yr", 365),
        ]

        if 'selected_relative' not in st.session_state:
            st.session_state.selected_relative = None

        rel_cols = st.columns(len(relative_options))
        for i, (label, days) in enumerate(relative_options):
            with rel_cols[i]:
                if st.button(label, use_container_width=True):
                    st.session_state.selected_relative = days
                    st.session_state.selected_year = None

        # Set date range based on selection
        if st.session_state.selected_relative is not None:
            days = st.session_state.selected_relative
            rel_start = max(min_date, (max_date - pd.Timedelta(days=days)))
            default_range = (rel_start, max_date)
        elif st.session_state.selected_year is not None:
            year = st.session_state.selected_year
            year_start = max(min_date, pd.Timestamp(f'{year}-01-01').to_pydatetime())
            year_end = min(max_date, pd.Timestamp(f'{year}-12-31').to_pydatetime())
            default_range = (year_start, year_end)
        else:
            default_range = (min_date, max_date)

        date_range = st.slider(
            "Time range:",
            min_value=min_date,
            max_value=max_date,
            value=default_range,
            format="YYYY-MM-DD"
        )

        col1, col2 = st.columns(2)

        # Aave max leverage (always included)
        aave_max_L = int(L_aave)

        # Calculate Twyne max leverage
        twyne_max_L = int(L_max_global)

        with col1:
            bt_leverage_input = st.text_input(
                "Leverage levels (comma-separated):",
                value=f"2, {aave_max_L}, 20, {twyne_max_L}",
                help=f"Aave max ({aave_max_L}x) and Twyne max ({twyne_max_L}x) included"
            )
            try:
                bt_leverage_levels = [int(x.strip()) for x in bt_leverage_input.split(",") if x.strip()]
                # Always ensure Aave max leverage is included
                if aave_max_L not in bt_leverage_levels:
                    bt_leverage_levels.append(aave_max_L)
                bt_leverage_levels = sorted(set(bt_leverage_levels))
            except ValueError:
                bt_leverage_levels = [2, aave_max_L, 20, 30]
                st.warning("Invalid input, using defaults")

        with col2:
            bt_initial_hf = st.number_input(
                "Initial Health Factor:",
                min_value=1.0001, max_value=1.50, value=1.05, step=0.0001, format="%.4f",
                help="HF=1.0 means at liquidation threshold"
            )

    # Filter dataframe by date range
    df_filtered = df[(df.index >= date_range[0]) & (df.index <= date_range[1])]

    st.caption(f"{len(df_filtered)} days ({date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')})")

    # Auto-run backtest
    if bt_leverage_levels and len(df_filtered) > 0:
        results = run_multi_leverage_backtest(
            df_filtered,
            leverage_levels=bt_leverage_levels,
            initial_hf=bt_initial_hf,
            utilization=utilization,
            **params
        )

        # Results table
        st.markdown("**Results Summary**")
        st.caption("Key performance metrics for each leverage level. Progress bars show relative values. Hover column headers for metric explanations.")

        summary_data = []
        for L in sorted(results.keys()):
            r = results[L]
            name, _, _ = get_leverage_style(L)

            # Calculate volatility (annualized daily std)
            valid_pnl = r.pnl_daily[~np.isnan(r.pnl_daily)]
            volatility = np.std(valid_pnl) * np.sqrt(365) * 100 if len(valid_pnl) > 1 else 0

            summary_data.append({
                "Leverage": name,
                "Ann. Return": r.metrics['annualized_return'] * 100,
                "Max Drawdown": r.metrics['max_drawdown'] * 100,
                "Volatility": volatility,
                "Sharpe": r.metrics['sharpe_ratio'] if not np.isnan(r.metrics['sharpe_ratio']) else 0,
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("Ann. Return", ascending=False)

        # Configure columns with progress bars for visual comparison
        col_config = {
            "Leverage": st.column_config.TextColumn(
                "Leverage",
                width="small",
                help="Protocol and leverage level. Aave = no Twyne fees, Twyne = boosted leverage with credit delegation costs.",
            ),
            "Ann. Return": st.column_config.ProgressColumn(
                "Ann. Return",
                format="%.1f%%",
                min_value=min(0, summary_df["Ann. Return"].min()),
                max_value=summary_df["Ann. Return"].max(),
                help="Annualized return. What the total return would be if extrapolated to a full year.",
            ),
            "Max Drawdown": st.column_config.ProgressColumn(
                "Max Drawdown",
                format="%.1f%%",
                min_value=0,
                max_value=summary_df["Max Drawdown"].max(),
                help="Maximum drawdown. Largest peak-to-trough decline. Lower is better.",
            ),
            "Volatility": st.column_config.ProgressColumn(
                "Volatility",
                format="%.1f%%",
                min_value=0,
                max_value=summary_df["Volatility"].max(),
                help="Annualized volatility. Standard deviation of daily returns. Lower = smoother ride.",
            ),
            "Sharpe": st.column_config.ProgressColumn(
                "Sharpe",
                format="%.2f",
                min_value=min(0, summary_df["Sharpe"].min()),
                max_value=summary_df["Sharpe"].max(),
                help="Sharpe ratio (return ÷ volatility). Higher is better. Above 1.0 is generally good.",
            ),
        }

        st.dataframe(summary_df, column_config=col_config, hide_index=True, use_container_width=True)

        # -----------------------------------------------------------------
        # 1. Capital Evolution
        # -----------------------------------------------------------------
        st.markdown("**Capital Evolution**")
        st.caption("Growth of $1 invested at each leverage level. Higher leverage amplifies both gains and losses. Hover to see return % at any point.")
        fig_capital = go.Figure()

        # Add all results with proper naming/coloring
        for L in sorted(results.keys()):
            r = results[L]
            name, color, width = get_leverage_style(L)
            # Calculate return % for tooltip
            capital_return = (r.capital - 1) * 100
            fig_capital.add_trace(go.Scatter(
                x=r.dates, y=r.capital,
                name=name,
                mode='lines',
                line=dict(color=color, width=width),
                customdata=capital_return,
                hovertemplate="%{customdata:.1f}%<extra></extra>",
            ))

        fig_capital.add_hline(y=1.0, line_dash="dot", line_color="gray", line_width=1)
        fig_capital.update_layout(
            xaxis_title="Date",
            yaxis_title="Capital (starting at 1.0)",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            hovermode='x unified',
        )
        st.plotly_chart(fig_capital, width='stretch')

        # -----------------------------------------------------------------
        # 2. Rolling Annualized Yield
        # -----------------------------------------------------------------
        st.markdown("**Rolling Annualized Yield**")

        # Window period selector
        rolling_window_options = {"Daily": 1, "7 Days": 7, "30 Days": 30}
        rolling_window_labels = list(rolling_window_options.keys())
        rolling_window_col, _ = st.columns([1, 3])
        with rolling_window_col:
            selected_window_label = st.segmented_control(
                "Window",
                rolling_window_labels,
                default="30 Days",
                label_visibility="collapsed",
            )
        rolling_window = rolling_window_options[selected_window_label]

        window_label = "Daily" if rolling_window == 1 else f"{rolling_window}-day"
        st.caption(f"{window_label} return annualized to yearly rate. Shows yield volatility over time. Wide swings = inconsistent returns. Below 0% = losing money during that period.")
        fig_rolling = go.Figure()

        for L in sorted(results.keys()):
            r = results[L]
            name, color, width = get_leverage_style(L)
            if len(r.capital) >= rolling_window:
                capital_series = pd.Series(r.capital, index=r.dates)
                rolling_return = capital_series / capital_series.shift(rolling_window)
                # Annualize: (1 + window_return)^(365/window) - 1
                rolling_annualized = (rolling_return ** (365/rolling_window) - 1) * 100
                fig_rolling.add_trace(go.Scatter(
                    x=r.dates[rolling_window:], y=rolling_annualized.values[rolling_window:],
                    name=name,
                    mode='lines',
                    line=dict(color=color, width=width),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))

        fig_rolling.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
        fig_rolling.update_layout(
            xaxis_title="Date",
            yaxis_title="Annualized Yield (%)",
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            hovermode='x unified',
        )
        st.plotly_chart(fig_rolling, width='stretch')

        # -----------------------------------------------------------------
        # 3. Drawdown Visualization
        # -----------------------------------------------------------------
        st.markdown("**Drawdown**")
        st.caption("Percentage below all-time high at each point. Deepest dips = max drawdown (worst peak-to-trough loss). Higher leverage = deeper drawdowns.")
        fig_dd = go.Figure()

        for L in sorted(results.keys()):
            r = results[L]
            name, color, width = get_leverage_style(L)
            capital_series = pd.Series(r.capital, index=r.dates)
            rolling_max = capital_series.expanding().max()
            drawdown = (capital_series - rolling_max) / rolling_max * 100

            fig_dd.add_trace(go.Scatter(
                x=r.dates, y=drawdown.values,
                name=name,
                mode='lines',
                fill='tozeroy',
                line=dict(color=color, width=width),
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))

        fig_dd.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=300,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            hovermode='x unified',
        )
        st.plotly_chart(fig_dd, width='stretch')

        # -----------------------------------------------------------------
        # 4. Timing Sensitivity (Convergence Cone)
        # -----------------------------------------------------------------
        st.markdown("**Timing Sensitivity**")
        st.caption("Does it matter when you enter? Charts show return range and loss probability across all possible entry dates for each holding period.")

        # More granular holding periods for hover, but only show key labels on axis
        holding_periods = list(range(7, 366, 7))  # Every 7 days from 7 to 364
        holding_periods_labels = [7, 30, 90, 180, 365]  # Only show these on axis

        # Calculate return distribution using existing backtest capital data (FAST)
        sensitivity_data = {}
        for L in sorted(bt_leverage_levels):
            sensitivity_data[L] = {'best': [], 'worst': [], 'median': [], 'loss_pct': []}
            r = results[L]
            capital_series = pd.Series(r.capital, index=r.dates)

            for hold_days in holding_periods:
                if len(capital_series) > hold_days:
                    # Calculate returns for all possible entry dates at once
                    rolling_return = capital_series / capital_series.shift(hold_days) - 1
                    # Annualize for comparability
                    annualized = ((1 + rolling_return) ** (365 / hold_days) - 1) * 100
                    valid = annualized.dropna()

                    if len(valid) > 0:
                        sensitivity_data[L]['best'].append(valid.max())
                        sensitivity_data[L]['worst'].append(valid.min())
                        sensitivity_data[L]['median'].append(valid.median())
                        sensitivity_data[L]['loss_pct'].append((valid < 0).sum() / len(valid) * 100)
                    else:
                        sensitivity_data[L]['best'].append(np.nan)
                        sensitivity_data[L]['worst'].append(np.nan)
                        sensitivity_data[L]['median'].append(np.nan)
                        sensitivity_data[L]['loss_pct'].append(np.nan)
                else:
                    sensitivity_data[L]['best'].append(np.nan)
                    sensitivity_data[L]['worst'].append(np.nan)
                    sensitivity_data[L]['median'].append(np.nan)
                    sensitivity_data[L]['loss_pct'].append(np.nan)

        # Find last holding period where any leverage has loss > 0 (for x-axis range)
        last_nonzero_idx = 0
        for L in bt_leverage_levels:
            loss_data = sensitivity_data[L]['loss_pct']
            for i, val in enumerate(loss_data):
                if val is not None and not np.isnan(val) and val > 0:
                    last_nonzero_idx = max(last_nonzero_idx, i)

        # Add some padding (20% more) after last non-zero point
        max_idx = min(len(holding_periods) - 1, int(last_nonzero_idx * 1.2) + 2)
        x_max = holding_periods[max_idx]
        filtered_labels = [d for d in holding_periods_labels if d <= x_max]

        # Create side-by-side timing sensitivity charts
        ts_col1, ts_col2 = st.columns(2)

        with ts_col1:
            st.markdown("**Return Range by Holding Period**")
            st.caption("Each band = one strategy's best-to-worst range. Higher leverage = wider band. White gaps = strategies don't overlap there.")
            # Convergence cone chart
            fig_cone = go.Figure()

            for L in sorted(bt_leverage_levels):
                name, color, _ = get_leverage_style(L)
                data = sensitivity_data[L]

                # Shaded area between best and worst
                # Convert hex color to rgba with transparency
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    fill_color = f'rgba({r},{g},{b},0.2)'
                else:
                    fill_color = 'rgba(128,128,128,0.2)'

                fig_cone.add_trace(go.Scatter(
                    x=holding_periods + holding_periods[::-1],
                    y=data['best'] + data['worst'][::-1],
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{name} range',
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo='skip',
                ))

                # Median line
                fig_cone.add_trace(go.Scatter(
                    x=holding_periods,
                    y=data['median'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=name,
                    legendgroup=name,
                    hovertemplate=f"{name}<br>Holding Period: %{{x}} days<br>Median: %{{y:.1f}}%<br>Best: %{{customdata[0]:.1f}}%<br>Worst: %{{customdata[1]:.1f}}%<extra></extra>",
                    customdata=list(zip(data['best'], data['worst'])),
                ))

            fig_cone.update_layout(
                xaxis_title="Holding Period (days)",
                yaxis_title="Annualized Return (%)",
                height=350,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
                hovermode='x unified',
                xaxis=dict(
                    tickmode='array',
                    tickvals=[7, 30, 90, 180, 365],
                    ticktext=['7', '30', '90', '180', '365'],
                ),
            )
            fig_cone.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5)
            st.plotly_chart(fig_cone, use_container_width=True)

        with ts_col2:
            st.markdown("**Loss Probability by Holding Period**")
            st.caption("% of entry dates that lost money. Drops over time = longer holds historically safer.")
            # Loss probability chart
            fig_loss = go.Figure()

            for L in sorted(bt_leverage_levels):
                name, color, _ = get_leverage_style(L)
                data = sensitivity_data[L]

                fig_loss.add_trace(go.Scatter(
                    x=holding_periods,
                    y=data['loss_pct'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=name,
                    hovertemplate=f"{name}<br>Holding Period: %{{x}} days<br>Loss prob: %{{y:.1f}}%<extra></extra>",
                ))

            fig_loss.update_layout(
                xaxis_title="Holding Period (days)",
                yaxis_title="Loss Probability (%)",
                height=350,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
                hovermode='x unified',
                xaxis=dict(
                    tickmode='array',
                    tickvals=filtered_labels,
                    ticktext=[str(d) for d in filtered_labels],
                    range=[0, x_max],
                ),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # -----------------------------------------------------------------
        # 5. Risk-Adjusted Comparison (Bar Chart)
        # -----------------------------------------------------------------
        st.markdown("**Risk-Adjusted Performance**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sharpe Ratio**")
            st.caption("Return divided by volatility. Above 1.0 = good, above 2.0 = excellent. Measures return per unit of daily volatility.")
            # Sharpe Ratio comparison
            leverages = sorted(results.keys())
            sharpes = [results[L].metrics['sharpe_ratio'] if not np.isnan(results[L].metrics['sharpe_ratio']) else 0 for L in leverages]

            # Get colors and names using helper
            bar_colors = []
            bar_names = []
            bar_widths = []
            for L in leverages:
                name, color, _ = get_leverage_style(L)
                bar_names.append(name)
                bar_colors.append(color)
                bar_widths.append(0.6 if L <= int(L_aave) else 0.8)

            fig_sharpe = go.Figure(data=[
                go.Bar(
                    x=bar_names,
                    y=sharpes,
                    marker_color=bar_colors,
                    width=bar_widths,
                    text=[f'{s:.2f}' for s in sharpes],
                    textposition='outside',
                )
            ])
            # Add headroom for text labels above bars
            max_sharpe = max(sharpes) if sharpes else 1
            fig_sharpe.update_layout(
                xaxis_title="",
                yaxis_title="Sharpe Ratio",
                height=320,
                yaxis=dict(range=[0, max_sharpe * 1.15]),  # 15% headroom
            )
            st.plotly_chart(fig_sharpe, width='stretch')

        with col2:
            st.markdown("**Return vs Risk**")
            st.caption("Each point shows leverage's return (y) vs drawdown (x). Ideal = top-left (high return, low drawdown). Lines from origin show risk/return slope.")
            # Return vs Max Drawdown scatter
            returns = [results[L].metrics['annualized_return'] * 100 for L in leverages]
            drawdowns = [abs(results[L].metrics['max_drawdown']) * 100 for L in leverages]

            fig_risk = go.Figure()

            # Draw lines from origin to each point
            for i, L in enumerate(leverages):
                name, color, _ = get_leverage_style(L)
                if drawdowns[i] > 0:
                    fig_risk.add_trace(go.Scatter(
                        x=[0, drawdowns[i]],
                        y=[0, returns[i]],
                        mode='lines',
                        line=dict(color=color, width=1.5, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip',
                    ))

            # Add points on top
            for i, L in enumerate(leverages):
                name, color, _ = get_leverage_style(L)
                size = 12 if L <= int(L_aave) else 20
                fig_risk.add_trace(go.Scatter(
                    x=[drawdowns[i]],
                    y=[returns[i]],
                    mode='markers+text',
                    text=[name],
                    textposition='middle left',
                    marker=dict(size=size, color=color),
                    name=name,
                    showlegend=False,
                    hovertemplate=f"{name}<br>Return: %{{y:.1f}}%<br>Drawdown: %{{x:.1f}}%<extra></extra>",
                ))

            # Add padding to axes for labels
            max_dd = max(drawdowns) if drawdowns else 1
            min_dd = min(drawdowns) if drawdowns else 0
            max_ret = max(returns) if returns else 1
            fig_risk.update_layout(
                xaxis_title="Max Drawdown (%)",
                yaxis_title="Annualized Return (%)",
                height=320,
                xaxis=dict(range=[min_dd - max_dd * 0.4, max_dd * 1.3]),
                yaxis=dict(range=[0, max_ret * 1.2]),
                margin=dict(r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_risk, width='stretch')

    # =========================================================================
    # Historical Analysis Section
    # =========================================================================
    st.markdown("---")
    st.markdown("### Historical Analysis")
    st.caption("Raw historical rate data from Aave and Lido. Understanding past spread behavior helps assess strategy viability.")

    # Summary stats
    stats = get_summary_stats(df_filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Date Range", f"{stats['date_range']['n_days']} days")
    col2.metric("Mean Spread", f"{stats['spread']['mean']*100:.2f}%")
    col3.metric("Days Positive", f"{stats['spread']['pct_positive']:.1f}%")
    col4.metric("Min Spread", f"{stats['spread']['min']*100:.2f}%")

    # Historical rates plot (3 rows: Rates, Spread, Yield by Leverage)
    st.markdown("**Historical Rates & Yields**")
    st.caption("Top: stETH yield (green) vs ETH borrow cost (red). Middle: raw spread between them. Bottom: what each leverage level would have earned daily.")

    fig_hist = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.35, 0.25, 0.4],
                             subplot_titles=("Rates", "Spread", "Net Yield by Leverage"),
                             vertical_spacing=0.08)

    # Row 1: Rates
    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['r_stake'] * 100,
        name='stETH Yield', line=dict(color='#28A745'),
    ), row=1, col=1)

    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['r_borrow'] * 100,
        name='ETH Borrow', line=dict(color='#DC3545'),
    ), row=1, col=1)

    # Row 2: Spread
    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['spread_raw'] * 100,
        name='Raw Spread', fill='tozeroy',
        line=dict(color='#2E86AB'),
        showlegend=False,
    ), row=2, col=1)

    fig_hist.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="black")

    # Row 3: Yield by Leverage
    if bt_leverage_levels:
        for L in sorted(bt_leverage_levels):
            name, color, width = get_leverage_style(L)
            yields = []
            for _, row in df_filtered.iterrows():
                lt = ltv_from_leverage(L)
                llt = lt * 1.05  # Assume HF=1.05
                p = psi(llt, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
                y = looped_yield_from_leverage(row['r_stake'], row['r_borrow'], L, ir_u, p)
                yields.append(y * 100)

            fig_hist.add_trace(go.Scatter(
                x=df_filtered.index, y=yields,
                name=name,
                line=dict(color=color, width=width),
            ), row=3, col=1)

        fig_hist.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="black")

    fig_hist.update_layout(height=800, showlegend=True,
                           legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
                           hovermode='x unified')
    fig_hist.update_yaxes(title_text="APR (%)", row=1, col=1)
    fig_hist.update_yaxes(title_text="Spread (%)", row=2, col=1)
    fig_hist.update_yaxes(title_text="Net Yield (%)", row=3, col=1)

    st.plotly_chart(fig_hist, width='stretch')

    # =========================================================================
    # Spread Risk Analysis (Tail Risk & Worst Periods)
    # =========================================================================
    st.markdown("**Spread Risk Analysis**")
    st.caption("Tail risk metrics showing how bad things can get. Important for stress-testing strategy resilience.")

    # Calculate tail risk metrics
    spread_data = df_filtered['spread_raw'] * 100  # Convert to %
    spread_p5 = spread_data.quantile(0.05)
    spread_min = spread_data.min()
    pct_negative_spread = (spread_data < 0).mean() * 100
    pct_severe = (spread_data < -1).mean() * 100

    # 1. Tail Risk Metrics (2 columns)
    col1, col2 = st.columns(2)
    col1.metric("Worst spread ever", f"{spread_min:.2f}%")
    col2.metric("Days with negative spread", f"{pct_negative_spread:.1f}%")

    # 2. Rolling Return Analysis with window toggle
    st.markdown("**Rolling Return Analysis**")

    roll_window_col, _ = st.columns([1, 3])
    with roll_window_col:
        roll_window_label = st.segmented_control(
            "Window",
            ["1d", "7d", "30d"],
            default="30d",
            label_visibility="collapsed",
            key="roll_return_window"
        )

    roll_window = {"1d": 1, "7d": 7, "30d": 30}[roll_window_label]
    st.caption(f"Statistics for all possible {roll_window}-day holding windows.")

    if bt_leverage_levels and len(df_filtered) >= roll_window:
        # Calculate rolling returns for each leverage
        rolling_data = {}
        summary_rows = []

        for L in sorted(bt_leverage_levels):
            result = results.get(L) if 'results' in dir() else None
            if result is not None and len(result.capital) >= roll_window:
                capital_series = pd.Series(result.capital, index=result.dates)
                rolling_ret = (capital_series / capital_series.shift(roll_window) - 1) * 100
                rolling_ret = rolling_ret.dropna()
                rolling_data[L] = rolling_ret

                # Calculate stats
                pct_negative = (rolling_ret < 0).mean() * 100
                worst = rolling_ret.min()
                median = rolling_ret.median()
                best = rolling_ret.max()

                name, color, _ = get_leverage_style(L)
                summary_rows.append({
                    'Leverage': name,
                    'Worst Return': f"{worst:.1f}%",
                    'Median Return': f"{median:.1f}%",
                    'Best Return': f"{best:.1f}%",
                    '% Periods Lost': f"{pct_negative:.1f}%",
                })

        if summary_rows:
            # Side by side: distribution chart (left) and table (right)
            chart_col, table_col = st.columns(2)

            with chart_col:
                fig_dist_returns = go.Figure()

                # Calculate common bin range across all data
                all_returns = pd.concat([rolling_data[L] for L in rolling_data.keys()])
                bin_min = all_returns.min()
                bin_max = all_returns.max()
                bin_size = (bin_max - bin_min) / 30

                for L in sorted(rolling_data.keys()):
                    name, color, _ = get_leverage_style(L)
                    fig_dist_returns.add_trace(go.Histogram(
                        x=rolling_data[L],
                        name=name,
                        opacity=0.6,
                        marker_color=color,
                        xbins=dict(start=bin_min, end=bin_max, size=bin_size),
                    ))

                fig_dist_returns.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
                fig_dist_returns.update_layout(
                    xaxis_title=f"{roll_window_label} Return (%)",
                    yaxis_title="Frequency",
                    height=350,
                    barmode='overlay',
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    margin=dict(l=50, r=20, t=20, b=50),
                )
                st.plotly_chart(fig_dist_returns, use_container_width=True)

            with table_col:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Leverage': st.column_config.TextColumn(
                            'Leverage',
                            help='Strategy and leverage level'
                        ),
                        'Worst Return': st.column_config.TextColumn(
                            'Worst Return',
                            help=f'Lowest {roll_window_label} return across all holding periods'
                        ),
                        'Median Return': st.column_config.TextColumn(
                            'Median Return',
                            help=f'Typical (50th percentile) {roll_window_label} return'
                        ),
                        'Best Return': st.column_config.TextColumn(
                            'Best Return',
                            help=f'Highest {roll_window_label} return across all holding periods'
                        ),
                        '% Periods Lost': st.column_config.TextColumn(
                            '% Periods Lost',
                            help=f'Percentage of {roll_window_label} holding periods that lost money'
                        ),
                    }
                )

    else:
        st.caption(f"Run a backtest with at least {roll_window} days of data to see rolling return analysis.")

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("""
**Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results.
Always understand the risks before participating in DeFi.

Built with the Twyne stETH-ETH Looping Economics module.
""")
