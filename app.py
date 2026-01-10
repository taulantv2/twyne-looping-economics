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
# Live rates API removed - using manual inputs only

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

# Get params from session state first (needed for Rates calculations)
underlying_lltv = st.session_state.get("underlying_lltv", 93) / 100
liq_ltv_c = underlying_lltv
liq_ltv_clp = underlying_lltv
liq_ltv_max = st.session_state.get("twyne_liq_ltv_max", 97) / 100
utilization = st.session_state.get("twyne_utilization", 95) / 100
beta_safe = st.session_state.get("twyne_beta_safe", 1.00)
ir_I_min = st.session_state.get("ir_I_min", 0.0) / 100
ir_I_0 = st.session_state.get("ir_I_0", 0.8) / 100
ir_u_0 = st.session_state.get("ir_u_0", 90) / 100
ir_I_max = st.session_state.get("ir_I_max", 5.0) / 100
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

# Credit cost (Ψ × IR(u)) at max LLTV
psi_max = liq_ltv_max / (beta_safe * liq_ltv_clp) - liq_ltv_c / liq_ltv_clp
psi_max = max(psi_max, 0)

# CLP yield
clp_credit_yield = utilization * ir_u

# =============================================================================
# Rates (Expander) - At top with all derived values
# =============================================================================
with st.sidebar.expander("📊 Rates", expanded=True):
    r_steth_stake = st.number_input("stETH Staking %", min_value=0.0, value=3.5, step=0.5, key="r_steth_stake") / 100
    r_steth_lend = st.number_input("stETH Lending %", min_value=0.0, value=0.1, step=0.1, key="r_steth_lend") / 100
    r_borrow = st.number_input("ETH Borrow %", min_value=0.0, value=2.5, step=0.5, key="r_borrow") / 100
    r_stake = r_steth_stake + r_steth_lend
    spread = r_stake - r_borrow
    clp_total_yield = clp_credit_yield + r_stake

    st.caption(f"**Total:** {r_stake*100:.2f}% | **Spread:** {spread*100:.2f}% | **CLP Yield:** {clp_total_yield*100:.2f}%")

    st.divider()
    st.caption("**Net Looping Yield (Aave → Twyne)**")
    for hf in [1.10, 1.05, 1.01]:
        lambda_t_aave = liq_ltv_c / hf
        lambda_t_twyne = liq_ltv_max / hf
        L_aave = 1.0 / (1.0 - lambda_t_aave)
        L_twyne = 1.0 / (1.0 - lambda_t_twyne)
        aave_yield = (L_aave * r_stake - (L_aave - 1) * r_borrow) * 100
        twyne_yield = (L_twyne * r_stake - (L_twyne - 1) * r_borrow - L_twyne * psi_max * ir_u) * 100
        st.text(f"HF {hf:.2f}: {aave_yield:.1f}% ({L_aave:.0f}x) → {twyne_yield:.1f}% ({L_twyne:.0f}x)")

# =============================================================================
# LTVs & Safety Buffer (Expander)
# =============================================================================
with st.sidebar.expander("🎯 LTVs & Safety Buffer", expanded=False):
    st.slider(
        "Underlying LLTV %",
        min_value=70, max_value=96, value=93, step=1,
        help="Aave e-mode liquidation threshold (same for C and CLP)",
        key="underlying_lltv"
    )
    st.slider(
        "Twyne Max LLTV %",
        min_value=90, max_value=99, value=97, step=1,
        help="Twyne's maximum boosted liquidation LTV",
        key="twyne_liq_ltv_max"
    )
    st.slider(
        "Safety Buffer",
        min_value=0.85, max_value=1.00, value=1.00, step=0.01,
        help="Safety buffer for credit reservation",
        key="twyne_beta_safe"
    )

# =============================================================================
# Interest Rate Model (Expander) - includes CLP Utilization
# =============================================================================
with st.sidebar.expander("📈 Interest Rate Model", expanded=False):
    st.slider(
        "CLP Utilization %",
        min_value=0, max_value=100, value=95, step=5,
        help="Credit LP pool utilization - affects IR(u)",
        key="twyne_utilization"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Min Rate %", min_value=0.0, value=0.0, step=0.5, key="ir_I_min")
        st.number_input("Rate at Kink %", min_value=0.0, value=0.8, step=0.5, key="ir_I_0")
        st.number_input("Kink Utilization %", min_value=50.0, max_value=99.0, value=90.0, step=1.0, key="ir_u_0")
    with col2:
        st.number_input("Max Rate %", min_value=0.1, value=5.0, step=1.0, key="ir_I_max")
        st.number_input("Curvature", min_value=1, value=32, step=1, key="ir_gamma")

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

tab1, tab2, tab3 = st.tabs([
    "📈 Backtest & History",
    "📊 Yield Analysis",
    "⚖️ Liquidation Analysis"
])

# =============================================================================
# Tab 2: Yield Analysis (Detailed Sheet)
# =============================================================================

with tab2:
    st.header("Yield Analysis")
    st.caption("ℹ️ Uses: Rates, LTVs, Interest Rate Model")

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

    # Aave threshold in LTV terms (no Twyne boost needed below this)
    aave_lltv = liq_ltv_c  # underlying LLTV (e.g., 93%)

    # Use numeric x values (LTV as percentage number) for proper shape positioning
    x_values = liq_ltv_values * 100  # e.g., [92, 93.5, 95, 96.5, 98]
    y_values = hf_values  # e.g., [1.20, 1.18, ..., 1.00]

    # Compute display values based on mode
    yield_pct = yield_grid * 100

    if yield_mode == "Delta (vs Aave)":
        # Subtract the Aave baseline (first column) from each row
        aave_baseline = yield_pct[:, 0:1]  # First column (Aave max LTV)
        display_values = yield_pct - aave_baseline
        yield_title = "Extra Yield vs Aave"
        text_annotations = [[f"+{val:.1f}%" if val >= 0 else f"{val:.1f}%" for val in row] for row in display_values]
    else:
        display_values = yield_pct
        yield_title = "Net Yield Heatmap"
        text_annotations = [[f"{val:.1f}%" for val in row] for row in display_values]

    # Identify cells with positive yield (SAFE cells - no liquidation risk)
    safe_mask = (yield_grid > 0) | np.isinf(dtl_grid) | (dtl_grid > 9999)

    # Don't cap days - show actual values, but handle infinity
    dtl_display = dtl_grid.copy()
    dtl_display[np.isinf(dtl_display)] = np.nan  # Will be shown as SAFE
    dtl_display[safe_mask] = np.nan  # SAFE cells

    # Create text annotations for DTL: "SAFE" for positive yield/infinite, days for others
    dtl_text = []
    for i, row in enumerate(dtl_grid):
        text_row = []
        for j, val in enumerate(row):
            if safe_mask[i, j]:
                text_row.append("SAFE")
            elif val > 9999:
                text_row.append("SAFE")
            else:
                text_row.append(f"{int(val)}d")
        dtl_text.append(text_row)

    # Create masked arrays for separate plotting
    dtl_risk = dtl_display.copy().astype(float)
    dtl_risk[safe_mask] = np.nan
    dtl_safe = np.where(safe_mask, 1.0, np.nan)

    # Custom colorscale for yield - relative to holding stETH (r_stake)
    # Red = worse than holding, White = same as holding, Blue = better than holding
    min_val = np.nanmin(display_values)
    max_val = np.nanmax(display_values)
    hold_yield = r_stake * 100  # stETH holding yield in %

    # Calculate where hold_yield falls in the data range
    if max_val > min_val:
        if hold_yield <= min_val:
            # All values above holding - all blue gradient
            custom_colorscale = [
                [0.0, '#f0f9ff'],    # Very light blue (lowest, at holding)
                [0.25, '#bae6fd'],   # Light sky blue
                [0.5, '#7dd3fc'],    # Sky blue
                [0.75, '#38bdf8'],   # Bright blue
                [1.0, '#0284c7'],    # Strong blue (highest)
            ]
        elif hold_yield >= max_val:
            # All values below holding - all red gradient
            custom_colorscale = [
                [0.0, '#991b1b'],    # Dark red (most below holding)
                [0.25, '#dc2626'],   # Red
                [0.5, '#f87171'],    # Light red
                [0.75, '#fecaca'],   # Very light red
                [1.0, '#fff7f7'],    # Almost white (at holding)
            ]
        else:
            # Mixed: diverging red-white-blue centered at hold_yield
            hold_pos = (hold_yield - min_val) / (max_val - min_val)
            hold_pos = max(0.1, min(0.9, hold_pos))
            custom_colorscale = [
                [0.0, '#b91c1c'],                           # Red (most below holding)
                [hold_pos * 0.5, '#f87171'],                # Light red
                [hold_pos, '#ffffff'],                       # White (at holding yield)
                [hold_pos + (1-hold_pos) * 0.5, '#7dd3fc'], # Light blue
                [1.0, '#0284c7'],                           # Blue (most above holding)
            ]
    else:
        # All same value
        custom_colorscale = [[0.0, '#ffffff'], [1.0, '#ffffff']]

    # Create combined figure with subplots (linked axes for zoom/pan sync)
    fig_combined = make_subplots(
        rows=1, cols=2,
        subplot_titles=(yield_title, "Days to Liquidation"),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    # Left: Yield heatmap with explicit min/max for contrast
    fig_combined.add_trace(go.Heatmap(
        z=display_values,
        x=x_values,
        y=y_values,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 10, "color": "black"},
        colorscale=custom_colorscale,
        zmin=min_val,
        zmax=max_val,
        zmid=0 if min_val < 0 < max_val else None,
        showscale=False,
        hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br>Yield: %{z:.2f}%<extra></extra>",
    ), row=1, col=1)

    # Right: DTL risk heatmap (red-yellow-green)
    # Dynamic zmax based on actual non-safe values
    valid_dtl = dtl_risk[~np.isnan(dtl_risk)]
    dtl_zmax = max(365, np.nanmax(valid_dtl)) if len(valid_dtl) > 0 else 365
    fig_combined.add_trace(go.Heatmap(
        z=dtl_risk,
        x=x_values,
        y=y_values,
        colorscale='RdYlGn',
        zmin=0,
        zmax=dtl_zmax,
        hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br>Days: %{z:.0f}<extra></extra>",
        showscale=False,
    ), row=1, col=2)

    # Right: DTL safe heatmap (white for safe cells)
    fig_combined.add_trace(go.Heatmap(
        z=dtl_safe,
        x=x_values,
        y=y_values,
        colorscale=[[0, '#ffffff'], [1, '#ffffff']],
        zmin=0,
        zmax=1,
        hovertemplate="Liq LTV: %{x:.1f}%<br>HF: %{y:.2f}<br><b>SAFE</b><extra></extra>",
        showscale=False,
    ), row=1, col=2)

    # Add text annotations for DTL - only for non-safe cells
    for i, hf in enumerate(y_values):
        for j, ltv in enumerate(x_values):
            if not safe_mask[i, j]:
                fig_combined.add_annotation(
                    x=ltv,
                    y=hf,
                    text=f"{int(dtl_display[i, j])}d",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xref='x2',
                    yref='y2'
                )

    # Add single "SAFE" label in the center of safe region
    if np.any(safe_mask):
        safe_rows, safe_cols = np.where(safe_mask)
        center_row = int(np.mean(safe_rows))
        center_col = int(np.mean(safe_cols))
        fig_combined.add_annotation(
            x=x_values[center_col],
            y=y_values[center_row],
            text="<b>SAFE</b>",
            showarrow=False,
            font=dict(size=14, color="#006837"),
            xref='x2',
            yref='y2'
        )

    # Add Aave column border and label
    # Calculate column width (half distance to next column)
    col_width = (x_values[1] - x_values[0]) / 2 if len(x_values) > 1 else 0.5
    aave_x = x_values[0]  # First column is Aave
    hf_step = abs(y_values[1] - y_values[0]) if len(y_values) > 1 else 0.02

    for col_idx, xref in enumerate(['x', 'x2'], 1):
        yref = 'y' if col_idx == 1 else 'y2'
        # Add rectangle border around first column (within cell boundaries)
        fig_combined.add_shape(
            type="rect",
            x0=aave_x - col_width * 0.95,
            x1=aave_x + col_width * 0.95,
            y0=y_values[-1] - hf_step * 0.45,  # Bottom edge of lowest cell
            y1=y_values[0] + hf_step * 0.45,   # Top edge of highest cell
            line=dict(color="#ef6c00", width=4),
            fillcolor="rgba(0,0,0,0)",
            xref=xref,
            yref=yref,
            layer="above",
        )

    # Add "Aave" label above first column for both subplots (using paper coordinates)
    # Calculate x position in paper coordinates
    aave_rel_x = (aave_x - (x_values[0] - col_width)) / (x_values[-1] + col_width - (x_values[0] - col_width))

    # Left subplot (0 to ~0.48)
    fig_combined.add_annotation(
        x=aave_rel_x * 0.45,
        y=1.08,
        text="<b>Aave</b>",
        showarrow=False,
        font=dict(size=13, color="#ef6c00"),
        xref='paper',
        yref='paper',
    )

    # Right subplot (0.52 to 1.0)
    fig_combined.add_annotation(
        x=0.52 + aave_rel_x * 0.45,
        y=1.08,
        text="<b>Aave</b>",
        showarrow=False,
        font=dict(size=13, color="#ef6c00"),
        xref='paper',
        yref='paper',
    )

    fig_combined.update_layout(
        height=400,
        margin=dict(t=50, b=60),  # More top margin for label
        showlegend=False,
        hovermode='closest',
    )

    # Update axes
    fig_combined.update_xaxes(
        title_text="Boosted Liquidation LTV %",
        tickmode='array',
        tickvals=x_values,
        ticktext=[f"{v:.1f}" for v in x_values],
        row=1, col=1
    )
    fig_combined.update_xaxes(
        title_text="Boosted Liquidation LTV %",
        tickmode='array',
        tickvals=x_values,
        ticktext=[f"{v:.1f}" for v in x_values],
        row=1, col=2
    )
    fig_combined.update_yaxes(
        title_text="Health Factor",
        range=[y_values[-1] - hf_step * 0.5, y_values[0] + hf_step * 0.5],
        tickmode='array',
        tickvals=y_values,
        ticktext=[f"{v:.2f}" for v in y_values],
        row=1, col=1
    )
    fig_combined.update_yaxes(
        range=[y_values[-1] - hf_step * 0.5, y_values[0] + hf_step * 0.5],
        row=1, col=2
    )

    st.caption("Yield heatmap (left) and days to liquidation (right). Days to liquidation only applies when spread is negative. SAFE = positive yield, no liquidation risk.")
    st.plotly_chart(fig_combined, use_container_width=True)

    # =========================================================================
    # Twyne Credit Efficiency
    # =========================================================================
    # Calculate bounds for efficiency calculation
    L_aave = 1.0 / (1.0 - liq_ltv_c)  # ~14x for 93% e-mode
    aave_max_int = int(L_aave)
    twyne_max_int = int(L_max_global)

    # HF values for multi-line charts
    hf_values = [1.07, 1.10]
    hf_colors = {1.07: '#4caf50', 1.10: '#2196f3'}
    ref_hf = 1.05  # Default reference HF for single-HF calculations

    # Helper to calculate yield for a leverage level at a given HF
    def calc_yield_for_leverage_hf(L, r_stk, r_brw, hf):
        """Calculate net yield for leverage L at given rates and HF."""
        is_twyne = L > L_aave
        if is_twyne:
            lt = (L - 1) / L
            llt_needed = min(lt * hf, liq_ltv_max)
            p = max(0, psi(llt_needed, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c']))
            credit_cost = L * p * ir_u
        else:
            credit_cost = 0
        return (L * r_stk - (L - 1) * r_brw - credit_cost) * 100

    def calc_yield_for_leverage(L, r_stk, r_brw):
        return calc_yield_for_leverage_hf(L, r_stk, r_brw, ref_hf)

    st.markdown("**Twyne Credit Efficiency**")
    st.caption("How yield components change as you increase leverage. ℹ️ Uses: Rates, LTVs, IR Model")

    aave_L = aave_max_int
    aave_max_yield = calc_yield_for_leverage(aave_L, r_stake, r_borrow)

    # Generate data for all leverage levels from 1x to Twyne max (for waterfall)
    leverage_range = np.arange(1, twyne_max_int + 1)
    all_data = {}

    for L in leverage_range:
        L_yield = calc_yield_for_leverage(L, r_stake, r_borrow)

        if L > aave_L:
            extra_leverage = L - aave_L
            lt = (L - 1) / L
            llt_needed = min(lt * ref_hf, liq_ltv_max)
            p = max(0, psi(llt_needed, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c']))
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

    twyne_data = all_data

    leverages = list(leverage_range)
    aave_leverages = [L for L in leverages if L <= aave_max_int]
    twyne_leverages = [L for L in leverages if L >= aave_max_int]
    aave_yields = [calc_yield_for_leverage_hf(L, r_stake, r_borrow, 1.0) for L in aave_leverages]
    aave_max_yield_val = aave_yields[-1]  # Yield at Aave max leverage

    # =========================================================================
    # Yield vs Leverage + Waterfall Chart (side by side)
    # =========================================================================
    # Max HF where Twyne leverage > Aave leverage: HF < liq_ltv_max / liq_ltv_c
    max_hf_for_twyne = liq_ltv_max / liq_ltv_c  # e.g., 97%/93% ≈ 1.043

    # HF slider above both charts - controls the Twyne yield curve slope
    slider_hf = st.slider(
        "Health Factor",
        min_value=1.0,
        max_value=round(max_hf_for_twyne, 2),
        value=1.0,
        step=0.01,
        format="%.2f",
        key="yield_hf_slider",
        help=f"Max HF for Twyne leverage > Aave: {max_hf_for_twyne:.2f}"
    )

    # Calculate max leverage at selected HF
    slider_lt = liq_ltv_max / slider_hf
    if slider_lt >= 1:
        slider_max_L = twyne_max_int
    else:
        slider_max_L = min(int(1 / (1 - slider_lt)), twyne_max_int)
    slider_leverages = [L for L in twyne_leverages if L <= slider_max_L]
    slider_yields = [calc_yield_for_leverage_hf(L, r_stake, r_borrow, slider_hf) for L in slider_leverages]
    slider_gross = [(L * r_stake - (L - 1) * r_borrow) * 100 for L in slider_leverages]

    # Create yield vs leverage chart
    fig_eff = go.Figure()

    # Aave yield line (1x to Aave max)
    fig_eff.add_trace(go.Scatter(
        x=aave_leverages,
        y=aave_yields,
        mode='lines',
        name='Aave Yield',
        line=dict(color='#ef6c00', width=3),
        hovertemplate='Aave %{x}x: %{y:.1f}%<extra></extra>',
        legendrank=1,
    ))

    # Gross yield line (no credit cost) - dotted
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=slider_gross,
        mode='lines',
        name='Gross (no credit cost)',
        line=dict(color='#9e9e9e', width=2, dash='dot'),
        hovertemplate='Gross %{x}x: %{y:.1f}%<extra></extra>',
        legendrank=4,
    ))

    # Red fill: Gross yield (top) to Twyne yield (bottom) = credit cost
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=slider_gross,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=slider_yields,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor='rgba(139, 0, 0, 0.35)',
        showlegend=False,
        hoverinfo='skip',
    ))

    # Fill: Aave max yield (bottom) to Twyne yield (top)
    # Green for yield boost (positive spread), red for yield loss (negative spread)
    twyne_extra_color = 'rgba(76, 175, 80, 0.25)' if (r_stake - r_borrow) >= 0 else 'rgba(198, 40, 40, 0.25)'
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=[aave_max_yield_val] * len(slider_leverages),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=slider_yields,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor=twyne_extra_color,
        showlegend=False,
        hoverinfo='skip',
    ))
    # Twyne yield line (legend entry without fill)
    fig_eff.add_trace(go.Scatter(
        x=slider_leverages,
        y=slider_yields,
        mode='lines',
        name='Twyne Yield',
        line=dict(color='#9c27b0', width=3),
        hovertemplate=f'Twyne<br>%{{x}}x: %{{y:.1f}}%<extra></extra>',
        legendrank=2,
    ))

    # Flat dotted line for Aave max yield
    fig_eff.add_trace(go.Scatter(
        x=[aave_max_int, slider_max_L],
        y=[aave_max_yield_val, aave_max_yield_val],
        mode='lines',
        name='Aave max yield',
        line=dict(color='#ef6c00', width=2, dash='dot'),
        hovertemplate=f'Aave max yield: {aave_max_yield_val:.1f}%<extra></extra>',
        legendrank=3,
    ))

    # Legend entries for shaded areas
    fig_eff.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name='Credit cost',
        marker=dict(symbol='square', size=15, color='rgba(139, 0, 0, 0.35)'),
        legendrank=5,
    ))
    # Dynamic label and color: "yield boost" (green) when positive spread, "yield loss" (red) when negative
    twyne_extra_label = 'Twyne yield boost' if (r_stake - r_borrow) >= 0 else 'Twyne yield loss'
    fig_eff.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', name=twyne_extra_label,
        marker=dict(symbol='square', size=15, color=twyne_extra_color),
        legendrank=6,
    ))

    # Vertical line at Aave max
    fig_eff.add_vline(
        x=aave_max_int, line_dash="dash", line_color="gray", line_width=1,
        annotation_text=f"Aave max ({aave_max_int}x)", annotation_position="top",
    )

    fig_eff.update_layout(
        title=f'Yield vs Leverage (HF {slider_hf:.2f})',
        xaxis_title='Leverage (x)',
        yaxis_title='Net Yield (%)',
        height=380,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, traceorder='grouped'),
        margin=dict(l=50, r=150, t=40, b=50),
        hovermode='x unified',
    )

    yield_col, waterfall_col = st.columns([2, 1])

    with yield_col:
        st.plotly_chart(fig_eff, use_container_width=True)

    # Calculate yield components at slider HF for waterfall
    wf_psi = psi(liq_ltv_max, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])
    wf_gross = slider_max_L * r_stake - (slider_max_L - 1) * r_borrow
    wf_credit_cost = slider_max_L * wf_psi * ir_u
    wf_net_yield = (wf_gross - wf_credit_cost) * 100
    wf_credit_cost_pct = wf_credit_cost * 100
    wf_extra_leverage = slider_max_L - aave_max_int
    wf_gross_extra = wf_extra_leverage * (r_stake - r_borrow) * 100

    # Create waterfall chart using Bar with manual positioning for custom colors
    # Dynamic labels based on whether extra leverage helps or hurts
    extra_label = 'Extra Yield' if wf_gross_extra >= 0 else 'Extra Loss'
    extra_text = f'+{wf_gross_extra:.1f}%' if wf_gross_extra >= 0 else f'{wf_gross_extra:.1f}%'

    # Determine colors: Aave=orange, Extra=red/green based on direction, Credit=dark red, Twyne=purple
    extra_color = '#4CAF50' if wf_gross_extra >= 0 else '#c62828'

    # Calculate waterfall positions manually
    # Bar 1: Aave (starts at 0)
    # Bar 2: Extra (starts at top of Aave)
    # Bar 3: Credit Cost (starts at top of Extra, goes down)
    # Bar 4: Twyne total (starts at 0)
    x_labels = [f'Aave {aave_max_int}x', extra_label, 'Credit Cost', f'Twyne {slider_max_L}x']

    # Calculate bases and heights for waterfall effect
    aave_base = 0 if aave_max_yield >= 0 else aave_max_yield
    aave_height = abs(aave_max_yield)

    extra_base = aave_max_yield if wf_gross_extra >= 0 else aave_max_yield + wf_gross_extra
    extra_height = abs(wf_gross_extra)

    credit_top = aave_max_yield + wf_gross_extra
    credit_base = credit_top - wf_credit_cost_pct if wf_credit_cost_pct >= 0 else credit_top
    credit_height = abs(wf_credit_cost_pct)

    twyne_base = 0 if wf_net_yield >= 0 else wf_net_yield
    twyne_height = abs(wf_net_yield)

    fig_waterfall = go.Figure()

    # Add bars individually for custom colors
    bar_data = [
        (x_labels[0], aave_base, aave_height, '#ef6c00', f'{aave_max_yield:.1f}%'),
        (x_labels[1], extra_base, extra_height, extra_color, extra_text),
        (x_labels[2], credit_base, credit_height, '#8b0000', f'-{wf_credit_cost_pct:.1f}%'),
        (x_labels[3], twyne_base, twyne_height, '#7b1fa2', f'{wf_net_yield:.1f}%'),
    ]

    for x, base, height, color, text in bar_data:
        fig_waterfall.add_trace(go.Bar(
            x=[x],
            y=[height],
            base=[base],
            marker_color=color,
            text=[text],
            textposition='outside',
            textfont=dict(size=14),
            showlegend=False,
        ))

    # Dynamic y-axis range to handle negative values
    all_values = [aave_max_yield, aave_max_yield + wf_gross_extra,
                  aave_max_yield + wf_gross_extra - wf_credit_cost_pct, wf_net_yield]
    y_min = min(0, min(all_values) * 1.2)
    y_max = max(0, max(all_values) * 1.2)
    # Ensure some range if all values are near zero
    if y_max - y_min < 1:
        y_min, y_max = -1, 1

    fig_waterfall.update_layout(
        title=f'Yield Breakdown (HF {slider_hf:.2f})',
        height=380,
        yaxis_title='Annualized Yield (%)',
        showlegend=False,
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(categoryorder='array', categoryarray=x_labels),
        margin=dict(l=50, r=30, t=40, b=50),
        bargap=0.3,
    )

    with waterfall_col:
        st.plotly_chart(fig_waterfall, use_container_width=True)

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
        is_twyne = L > L_aave

        if is_twyne:
            lt = (L - 1) / L
            llt_needed = min(lt * ref_hf, liq_ltv_max)
            p = max(0, psi(llt_needed, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c']))
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

    # Spread Sensitivity chart only
    st.markdown("**Spread Sensitivity**")
    st.caption("How yield changes with spread.")
    st.plotly_chart(fig_sensitivity, use_container_width=True)

# =============================================================================
# Tab 1: Backtest & History
# =============================================================================

with tab1:
    # =========================================================================
    # Backtest Section
    # =========================================================================
    st.header("Historical Backtest")
    st.caption("ℹ️ Uses: LTVs, IR Model (historical rates from data)")

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
                max_value=max(0.1, summary_df["Ann. Return"].max()),  # Ensure non-zero max
                help="Annualized return. What the total return would be if extrapolated to a full year.",
            ),
            "Max Drawdown": st.column_config.ProgressColumn(
                "Max Drawdown",
                format="%.1f%%",
                min_value=0,
                max_value=max(0.1, summary_df["Max Drawdown"].max()),  # Ensure non-zero max
                help="Maximum drawdown. Largest peak-to-trough decline. Lower is better.",
            ),
            "Volatility": st.column_config.ProgressColumn(
                "Volatility",
                format="%.1f%%",
                min_value=0,
                max_value=max(0.1, summary_df["Volatility"].max()),  # Ensure non-zero max
                help="Annualized volatility. Standard deviation of daily returns. Lower = smoother ride.",
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

        # Window period selector - default based on data length
        rolling_window_options = {"Daily": 1, "7 Days": 7, "30 Days": 30}
        rolling_window_labels = list(rolling_window_options.keys())
        # Auto-select appropriate default based on data length
        if len(df_filtered) <= 14:
            default_window = "Daily"
        elif len(df_filtered) <= 45:
            default_window = "7 Days"
        else:
            default_window = "30 Days"

        # Force update if data length changed significantly
        bt_data_bucket = "short" if len(df_filtered) <= 14 else ("medium" if len(df_filtered) <= 45 else "long")
        if "bt_rolling_data_bucket" not in st.session_state or st.session_state.bt_rolling_data_bucket != bt_data_bucket:
            st.session_state.bt_rolling_data_bucket = bt_data_bucket
            st.session_state.bt_rolling_window = default_window

        rolling_window_col, _ = st.columns([1, 3])
        with rolling_window_col:
            selected_window_label = st.segmented_control(
                "Window",
                rolling_window_labels,
                default=st.session_state.get("bt_rolling_window", default_window),
                label_visibility="collapsed",
                key="bt_rolling_window"
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

        # Check if all drawdowns are essentially zero
        max_dd_all = max(r.metrics['max_drawdown'] for r in results.values())

        if max_dd_all < 0.0001:  # Essentially zero drawdown
            st.info("😎 **Up only!** No drawdowns during this period - all strategies stayed at or above their starting value.")
        else:
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

        # Dynamic holding periods based on data length
        max_data_days = len(df_filtered)
        if max_data_days <= 2:
            # Not enough data for timing sensitivity
            st.info("Need at least 3 days of data for timing sensitivity analysis.")
            holding_periods = []
            holding_periods_labels = []
        elif max_data_days <= 14:
            # Short period: daily increments, start from 1 day
            holding_periods = list(range(1, max_data_days))
            holding_periods_labels = [d for d in [1, 2, 3, 5, 7, 10, 14] if d < max_data_days]
        elif max_data_days <= 60:
            # Medium period: mix of daily and weekly
            holding_periods = list(range(1, 8)) + list(range(7, min(max_data_days, 61), 7))
            holding_periods = sorted(set(holding_periods))
            holding_periods_labels = [d for d in [1, 7, 14, 30, 60] if d < max_data_days]
        else:
            # Long period: weekly increments
            holding_periods = list(range(1, 8)) + list(range(7, min(max_data_days, 366), 7))
            holding_periods = sorted(set(holding_periods))
            holding_periods_labels = [d for d in [1, 7, 30, 90, 180, 365] if d < max_data_days]

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

        # Only show charts if we have holding periods data
        if holding_periods:
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
                        tickvals=filtered_labels,
                        ticktext=[str(d) for d in filtered_labels],
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
        # 5. Return vs Risk (Continuous Graph)
        # -----------------------------------------------------------------
        st.markdown("**Return vs Risk**")
        st.caption("Continuous curve showing return (y) vs max drawdown (x) for all leverage levels. Ideal = top-left (high return, low risk).")

        # Check if all drawdowns are essentially zero (reuse from earlier check)
        if max_dd_all < 0.0001:  # Essentially zero drawdown
            st.info("📈 **Up only!** No risk to plot - all strategies had zero drawdown during this period. Risk/return tradeoff chart not meaningful here.")
        else:
            # Compute returns and drawdowns for all leverage levels
            max_lev = int(L_max_global)
            all_leverages = list(range(2, max_lev + 1))

            all_returns = []
            all_drawdowns = []
            for L in all_leverages:
                if L in results:
                    ret = results[L].metrics['annualized_return'] * 100
                    dd = abs(results[L].metrics['max_drawdown']) * 100
                else:
                    try:
                        result = run_backtest(df_filtered, L, initial_hf=bt_initial_hf, utilization=utilization, **params)
                        ret = result.metrics['annualized_return'] * 100
                        dd = abs(result.metrics['max_drawdown']) * 100
                    except:
                        ret = 0
                        dd = 0
                all_returns.append(ret)
                all_drawdowns.append(dd)

            fig_risk = go.Figure()

            # Split into Aave and Twyne regions
            aave_levs = [L for L in all_leverages if L <= int(L_aave)]
            aave_returns = [all_returns[i] for i, L in enumerate(all_leverages) if L <= int(L_aave)]
            aave_drawdowns = [all_drawdowns[i] for i, L in enumerate(all_leverages) if L <= int(L_aave)]

            twyne_levs = [L for L in all_leverages if L > int(L_aave)]
            twyne_returns = [all_returns[i] for i, L in enumerate(all_leverages) if L > int(L_aave)]
            twyne_drawdowns = [all_drawdowns[i] for i, L in enumerate(all_leverages) if L > int(L_aave)]

            # Aave region line
            if aave_levs:
                fig_risk.add_trace(go.Scatter(
                    x=aave_drawdowns,
                    y=aave_returns,
                    mode='lines',
                    line=dict(color='#ef6c00', width=3),
                    name='Aave',
                    customdata=aave_levs,
                    hovertemplate='Aave %{customdata}x<br>Return: %{y:.1f}%<br>Drawdown: %{x:.1f}%<extra></extra>',
                ))

            # Twyne region line (connect from last Aave point)
            if twyne_levs:
                twyne_x = [aave_drawdowns[-1]] + twyne_drawdowns if aave_drawdowns else twyne_drawdowns
                twyne_y = [aave_returns[-1]] + twyne_returns if aave_returns else twyne_returns
                twyne_custom = [int(L_aave)] + twyne_levs if aave_levs else twyne_levs
                fig_risk.add_trace(go.Scatter(
                    x=twyne_x,
                    y=twyne_y,
                    mode='lines',
                    line=dict(color='#7b1fa2', width=3),
                    name='Twyne',
                    customdata=twyne_custom,
                    hovertemplate='Twyne %{customdata}x<br>Return: %{y:.1f}%<br>Drawdown: %{x:.1f}%<extra></extra>',
                ))

            # Add markers for selected leverage levels
            selected_leverages = sorted(results.keys())
            for L in selected_leverages:
                if L in all_leverages:
                    idx = all_leverages.index(L)
                    name, color, _ = get_leverage_style(L)
                    fig_risk.add_trace(go.Scatter(
                        x=[all_drawdowns[idx]],
                        y=[all_returns[idx]],
                        mode='markers+text',
                        marker=dict(size=14, color=color, line=dict(width=2, color='white')),
                        text=[f'{L}x'],
                        textposition='top center',
                        textfont=dict(size=10),
                        name=name,
                        showlegend=False,
                        hovertemplate=f'{name}<br>Return: %{{y:.1f}}%<br>Drawdown: %{{x:.1f}}%<extra></extra>',
                    ))

            max_dd = max(all_drawdowns) if all_drawdowns else 1
            max_ret = max(all_returns) if all_returns else 1
            fig_risk.update_layout(
                xaxis_title="Max Drawdown (%)",
                yaxis_title="Annualized Return (%)",
                height=350,
                xaxis=dict(range=[0, max_dd * 1.15]),
                yaxis=dict(range=[0, max_ret * 1.15]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

    # =========================================================================
    # Historical Analysis Section
    # =========================================================================
    st.markdown("---")
    st.markdown("### Historical Analysis")
    st.caption("Raw historical rate data from Aave and Lido. Understanding past spread behavior helps assess strategy viability.")

    # Summary stats
    stats = get_summary_stats(df_filtered)

    # Check if all spreads are positive (100% days positive)
    all_positive = stats['spread']['pct_positive'] >= 99.9

    if all_positive:
        st.info("📈 **All spreads positive!** The stETH yield exceeded ETH borrow cost every single day in this period.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Date Range", f"{stats['date_range']['n_days']} days")
    col2.metric("Mean Spread", f"{stats['spread']['mean']*100:.2f}%")
    col3.metric("Days Positive", f"{stats['spread']['pct_positive']:.1f}%")
    col4.metric("Min Spread", f"{stats['spread']['min']*100:.2f}%")
    col5.metric("Max Spread", f"{stats['spread']['max']*100:.2f}%")

    # Historical rates plot (2 rows: Rates+Spread combined, Yield by Leverage)
    st.markdown("**Historical Rates & Yields**")
    st.caption("Top: stETH yield (green), ETH borrow cost (red), and spread (blue filled). Bottom: what each leverage level would have earned daily.")

    fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.45, 0.55],
                             subplot_titles=("Rates & Spread", "Net Yield by Leverage"),
                             vertical_spacing=0.08)

    # Row 1: Rates + Spread on same axis (spread first so it's behind)
    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['spread_raw'] * 100,
        name='Spread', fill='tozeroy',
        line=dict(color='#2E86AB', width=1),
        fillcolor='rgba(46, 134, 171, 0.3)',
    ), row=1, col=1)

    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['r_stake'] * 100,
        name='stETH Yield', line=dict(color='#28A745', width=2),
    ), row=1, col=1)

    fig_hist.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered['r_borrow'] * 100,
        name='ETH Borrow', line=dict(color='#DC3545', width=2),
    ), row=1, col=1)

    fig_hist.add_hline(y=0, row=1, col=1, line_dash="dash", line_color="gray", line_width=0.5)

    # Row 2: Yield by Leverage
    if bt_leverage_levels:
        for L in sorted(bt_leverage_levels):
            name, color, width = get_leverage_style(L)
            yields = []
            # Calculate psi for this leverage (min LLTV needed at HF=1.05)
            lt = (L - 1) / L
            llt_needed = min(lt * 1.05, liq_ltv_max)
            p = max(0, psi(llt_needed, params['beta_safe'], params['liq_ltv_clp'], params['liq_ltv_c'])) if L > L_aave else 0
            for _, row in df_filtered.iterrows():
                y = looped_yield_from_leverage(row['r_stake'], row['r_borrow'], L, ir_u, p)
                yields.append(y * 100)

            fig_hist.add_trace(go.Scatter(
                x=df_filtered.index, y=yields,
                name=name,
                line=dict(color=color, width=width),
            ), row=2, col=1)

        fig_hist.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="black")

    fig_hist.update_layout(height=650, showlegend=True,
                           legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
                           hovermode='x unified')
    fig_hist.update_yaxes(title_text="APR (%)", row=1, col=1)
    fig_hist.update_yaxes(title_text="Net Yield (%)", row=2, col=1)

    st.plotly_chart(fig_hist, width='stretch')

    # =========================================================================
    # Spread Risk Analysis (Tail Risk & Worst Periods)
    # =========================================================================
    # Rolling Return Analysis with window toggle
    st.markdown("**Rolling Return Analysis**")

    # Auto-select appropriate default based on data length
    if len(df_filtered) <= 14:
        default_roll = "1d"
    elif len(df_filtered) <= 45:
        default_roll = "7d"
    else:
        default_roll = "30d"

    # Force update if data length changed significantly
    data_len_bucket = "short" if len(df_filtered) <= 14 else ("medium" if len(df_filtered) <= 45 else "long")
    if "roll_return_data_bucket" not in st.session_state or st.session_state.roll_return_data_bucket != data_len_bucket:
        st.session_state.roll_return_data_bucket = data_len_bucket
        st.session_state.roll_return_window = default_roll

    roll_window_col, _ = st.columns([1, 3])
    with roll_window_col:
        roll_window_label = st.segmented_control(
            "Window",
            ["1d", "7d", "30d"],
            default=st.session_state.get("roll_return_window", default_roll),
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
                # Use more decimals for small daily returns, fewer for longer periods
                decimals = 2 if roll_window <= 7 else 1
                summary_rows.append({
                    'Leverage': name,
                    'Worst Return': f"{worst:.{decimals}f}%",
                    'Median Return': f"{median:.{decimals}f}%",
                    'Best Return': f"{best:.{decimals}f}%",
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
# Tab 3: Time to Liquidation
# =============================================================================
with tab3:
    st.header("Time to Liquidation")
    st.caption("ℹ️ Uses: Rates, LTVs, IR Model")

    # =========================================================================
    # Chart: Aave vs Twyne - Survival by Health Factor (MOVED TO TOP)
    # =========================================================================
    st.markdown("**How Long Can You Survive a Negative Spread?**")
    st.caption("Compare Aave vs Twyne at different Health Factors. Lower HF = higher leverage = faster losses. Shaded bands show the range from mild (-1%) to severe (-5%) negative spreads.")

    # Spread bounds for analysis
    spread_best = -0.01
    spread_worst = -0.05

    # Ψ for Twyne (ir_u already calculated globally with sidebar params)
    twyne_psi = liq_ltv_max / (beta_safe * liq_ltv_clp) - liq_ltv_c / liq_ltv_clp
    twyne_psi = max(twyne_psi, 0)

    def compute_tliq_scalar(hf_val, r_net):
        decay = 1 + r_net / (1 + r_borrow)
        if r_net < 0 and decay > 0 and decay < 1:
            return min(-365 * np.log(hf_val) / np.log(decay), 1000)
        return 1000

    # HF range (1.01 to 1.10)
    hf_range_compare = np.linspace(1.01, 1.10, 80)

    # Implied leverage at each HF: L = HF / (HF - LLTV)
    aave_lev_from_hf = hf_range_compare / (hf_range_compare - liq_ltv_c)
    twyne_lev_from_hf = hf_range_compare / (hf_range_compare - liq_ltv_max)

    # Mask invalid (HF <= LLTV means infinite leverage)
    aave_hf_valid_mask = hf_range_compare > liq_ltv_c
    twyne_hf_valid_mask = hf_range_compare > liq_ltv_max

    aave_hf_valid = hf_range_compare[aave_hf_valid_mask]
    aave_lev_valid = aave_lev_from_hf[aave_hf_valid_mask]
    twyne_hf_valid = hf_range_compare[twyne_hf_valid_mask]
    twyne_lev_valid = twyne_lev_from_hf[twyne_hf_valid_mask]

    # Compute T_liq and loss for each HF
    aave_tliq_best_hf = np.array([compute_tliq_scalar(hf, spread_best) for hf in aave_hf_valid])
    aave_tliq_worst_hf = np.array([compute_tliq_scalar(hf, spread_worst) for hf in aave_hf_valid])
    aave_loss_best_hf = -(aave_lev_valid * spread_best + r_borrow) * 100 / 365
    aave_loss_worst_hf = -(aave_lev_valid * spread_worst + r_borrow) * 100 / 365

    twyne_tliq_best_hf = np.array([compute_tliq_scalar(hf, spread_best - twyne_psi * ir_u) for hf in twyne_hf_valid])
    twyne_tliq_worst_hf = np.array([compute_tliq_scalar(hf, spread_worst - twyne_psi * ir_u) for hf in twyne_hf_valid])
    twyne_loss_best_hf = -(twyne_lev_valid * spread_best - twyne_lev_valid * twyne_psi * ir_u + r_borrow) * 100 / 365
    twyne_loss_worst_hf = -(twyne_lev_valid * spread_worst - twyne_lev_valid * twyne_psi * ir_u + r_borrow) * 100 / 365

    fig_hf = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Time Until Liquidation (days)", "Daily Capital Loss (%)"),
        horizontal_spacing=0.12
    )

    # Aave band (orange)
    fig_hf.add_trace(go.Scatter(x=aave_hf_valid, y=aave_tliq_best_hf, mode='lines',
        line=dict(color='#ff9800', width=2.5),
        hovertemplate='Aave @ -1%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.0f} days<extra></extra>',
        customdata=aave_lev_valid, showlegend=False), row=1, col=1)
    fig_hf.add_trace(go.Scatter(x=aave_hf_valid, y=aave_tliq_worst_hf, mode='lines',
        line=dict(color='#ff9800', width=2.5, dash='dash'),
        hovertemplate='Aave @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.0f} days<extra></extra>',
        customdata=aave_lev_valid, fill='tonexty', fillcolor='rgba(255, 152, 0, 0.3)', showlegend=False), row=1, col=1)
    fig_hf.add_trace(go.Scatter(x=aave_hf_valid, y=-aave_loss_best_hf, mode='lines',
        line=dict(color='#ff9800', width=2.5),
        hovertemplate='Aave @ -1%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=aave_lev_valid, showlegend=False), row=1, col=2)
    fig_hf.add_trace(go.Scatter(x=aave_hf_valid, y=-aave_loss_worst_hf, mode='lines',
        line=dict(color='#ff9800', width=2.5, dash='dash'),
        hovertemplate='Aave @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=aave_lev_valid, fill='tonexty', fillcolor='rgba(255, 152, 0, 0.3)', showlegend=False), row=1, col=2)

    # Twyne band (blue)
    fig_hf.add_trace(go.Scatter(x=twyne_hf_valid, y=twyne_tliq_best_hf, mode='lines',
        line=dict(color='#2196f3', width=2),
        hovertemplate='Twyne @ -1%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.0f} days<extra></extra>',
        customdata=twyne_lev_valid, showlegend=False), row=1, col=1)
    fig_hf.add_trace(go.Scatter(x=twyne_hf_valid, y=twyne_tliq_worst_hf, mode='lines',
        line=dict(color='#2196f3', width=2, dash='dash'),
        hovertemplate='Twyne @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.0f} days<extra></extra>',
        customdata=twyne_lev_valid, fill='tonexty', fillcolor='rgba(33, 150, 243, 0.2)', showlegend=False), row=1, col=1)
    fig_hf.add_trace(go.Scatter(x=twyne_hf_valid, y=-twyne_loss_best_hf, mode='lines',
        line=dict(color='#2196f3', width=2),
        hovertemplate='Twyne @ -1%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=twyne_lev_valid, showlegend=False), row=1, col=2)
    fig_hf.add_trace(go.Scatter(x=twyne_hf_valid, y=-twyne_loss_worst_hf, mode='lines',
        line=dict(color='#2196f3', width=2, dash='dash'),
        hovertemplate='Twyne @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=twyne_lev_valid, fill='tonexty', fillcolor='rgba(33, 150, 243, 0.2)', showlegend=False), row=1, col=2)

    # Legend traces
    fig_hf.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Twyne', line=dict(color='#2196f3', width=3)))
    fig_hf.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Aave', line=dict(color='#ff9800', width=3)))
    fig_hf.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='-5% spread', line=dict(color='black', width=2, dash='dash')))
    fig_hf.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='-1% spread', line=dict(color='black', width=2)))

    fig_hf.update_layout(
        height=350, hovermode='x unified', margin=dict(l=60, t=80, r=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.18, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)')
    )
    fig_hf.update_yaxes(title_text="Days", row=1, col=1)
    fig_hf.update_yaxes(title_text="Daily Loss % (down=worse)", row=1, col=2)

    key_hf = [1.02, 1.04, 1.06, 1.08, 1.10]
    def make_tick(hf):
        aave_lev = hf / (hf - liq_ltv_c)
        twyne_lev = hf / (hf - liq_ltv_max)
        return f"{hf}<br><span style='color:#ff9800'>{aave_lev:.0f}x</span> / <span style='color:#2196f3'>{twyne_lev:.0f}x</span>"

    fig_hf.update_xaxes(title_text="Health Factor", row=1, col=1, tickvals=key_hf, ticktext=[make_tick(hf) for hf in key_hf])
    fig_hf.update_xaxes(title_text="Health Factor", row=1, col=2, tickvals=key_hf, ticktext=[make_tick(hf) for hf in key_hf])

    st.plotly_chart(fig_hf, use_container_width=True)

    # =========================================================================
    # Chart: HF vs Spread Sensitivity - Contour Heatmaps
    # =========================================================================
    st.markdown("---")
    st.markdown("**HF vs Spread Sensitivity: Aave vs Twyne**")
    st.caption("Contour maps showing days to liquidation. Green = safe (>365 days), Red = danger (<30 days).")

    # Grid for heatmap (HF limited to 1.01-1.05, negative spreads only)
    hf_range_sens = np.linspace(1.01, 1.05, 40)
    spread_range_sens = np.linspace(-0.05, 0.0, 50)  # -5% to 0%
    HF_grid, SPREAD_grid = np.meshgrid(hf_range_sens, spread_range_sens, indexing='ij')

    # Aave: r_net = spread (no Ψ×IR cost)
    aave_rnet = SPREAD_grid
    aave_decay = 1 + aave_rnet / (1 + r_borrow)
    aave_decay_safe = np.where(aave_decay > 0, aave_decay, 1)
    aave_tliq = np.where(
        aave_rnet >= 0, 500,
        np.where(aave_decay <= 0, 500,
                 np.clip(-365 * np.log(HF_grid) / np.log(aave_decay_safe), 0, 500))
    )

    # Twyne: r_net = spread - Ψ×IR (Ψ at max LLTV, using sidebar IR params)
    twyne_psi_val = liq_ltv_max / (beta_safe * liq_ltv_clp) - liq_ltv_c / liq_ltv_clp
    twyne_psi_val = max(twyne_psi_val, 0)
    # ir_u already calculated globally with sidebar params
    twyne_rnet = SPREAD_grid - twyne_psi_val * ir_u
    twyne_decay = 1 + twyne_rnet / (1 + r_borrow)
    twyne_decay_safe = np.where(twyne_decay > 0, twyne_decay, 1)
    twyne_tliq = np.where(
        twyne_rnet >= 0, 500,
        np.where(twyne_decay <= 0, 500,
                 np.clip(-365 * np.log(HF_grid) / np.log(twyne_decay_safe), 0, 500))
    )

    # Compute delta (Aave - Twyne): positive = Aave survives longer
    delta_tliq = aave_tliq - twyne_tliq

    # 2x2 grid: Aave heatmap + Delta heatmap on top, histograms below
    fig_sens = make_subplots(
        rows=2, cols=2,
        row_heights=[0.8, 0.2],
        subplot_titles=("Aave: Days to Liquidation", "Twyne Cost (extra days Aave survives)", "", ""),
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.12
    )

    # Row 1: Contour heatmaps
    # Dynamic zmax based on actual data
    aave_tliq_valid = aave_tliq[np.isfinite(aave_tliq)]
    aave_zmax = max(365, np.nanmax(aave_tliq_valid)) if len(aave_tliq_valid) > 0 else 365
    contour_end = min(aave_zmax, 1000)  # Cap contour labels at 1000 for readability
    fig_sens.add_trace(go.Contour(
        x=spread_range_sens * 100, y=hf_range_sens, z=aave_tliq,
        colorscale='RdYlGn', zmin=0, zmax=aave_zmax,
        contours=dict(showlines=True, showlabels=True, start=30, end=contour_end, size=max(60, contour_end/6)),
        hovertemplate='Spread: %{x:.1f}%<br>HF: %{y:.2f}<br>Days: %{z:.0f}<extra>Aave</extra>',
        showscale=False
    ), row=1, col=1)

    # Delta heatmap (shows extra days Aave survives vs Twyne)
    # White at 0, darker blue = more days lost
    # Dynamic range based on actual max delta
    delta_max = np.nanmax(delta_tliq[np.isfinite(delta_tliq)])
    delta_max = max(30, np.ceil(delta_max / 30) * 30)  # Round up to nearest 30, min 30
    contour_size = max(10, delta_max / 5)  # ~5 contour lines
    fig_sens.add_trace(go.Contour(
        x=spread_range_sens * 100, y=hf_range_sens, z=delta_tliq,
        colorscale=[[0, 'white'], [0.2, '#deebf7'], [0.4, '#9ecae1'], [0.6, '#4292c6'], [0.8, '#2171b5'], [1, '#084594']],
        zmin=0, zmax=delta_max,
        contours=dict(showlines=True, showlabels=True, labelfont=dict(size=11), start=0, end=delta_max, size=contour_size),
        hovertemplate='Spread: %{x:.1f}%<br>HF: %{y:.2f}<br>Twyne loses: %{z:.0f} days<extra></extra>',
        colorbar=dict(title='Days', x=1.02)
    ), row=1, col=2)

    # Row 2: Historical spread distribution (ALL data, not filtered, showing negative range -5% to 0%)
    spread_hist_all = df['spread_raw'].dropna() * 100  # Use full historical data, not filtered
    spread_negative = spread_hist_all[(spread_hist_all >= -5) & (spread_hist_all < 0)]

    # Calculate % of time in negative spread (all history)
    pct_negative = len(spread_hist_all[spread_hist_all < 0]) / len(spread_hist_all) * 100

    for col in [1, 2]:
        fig_sens.add_trace(go.Histogram(
            x=spread_negative,
            xbins=dict(start=-5, end=0, size=0.25),
            histnorm='percent',
            marker_color='rgba(100, 100, 100, 0.6)',
            showlegend=False,
            hovertemplate='Spread: %{x:.1f}%<br>Frequency: %{y:.1f}%<extra></extra>'
        ), row=2, col=col)

    # Add annotation showing % of time negative (with historical context)
    fig_sens.add_annotation(
        x=-4.5, y=0.9, xref="x3", yref="y3 domain",
        text=f"Negative {pct_negative:.0f}% of time (all history)",
        showarrow=False, font=dict(size=9, color="gray")
    )

    # Current spread marker (on all columns) - only if negative
    current_spread_pct = (r_stake - r_borrow) * 100
    if current_spread_pct < 0:
        for row in [1, 2]:
            for col in [1, 2]:
                # Add "Now" annotation on top row for both plots
                annotation = dict(text="Now", font_color="blue") if row == 1 else None
                fig_sens.add_vline(x=current_spread_pct, line_dash="dash", line_color="blue", row=row, col=col,
                                   annotation=annotation)

    fig_sens.update_layout(height=420, margin=dict(l=60, r=80, t=60, b=40))
    for col in [1, 2]:
        fig_sens.update_xaxes(title_text="Spread (%)", range=[-5, 0], row=2, col=col)
        fig_sens.update_xaxes(range=[-5, 0], row=1, col=col)
        fig_sens.update_yaxes(showticklabels=False, title_text="", row=2, col=col)
    fig_sens.update_yaxes(title_text="Health Factor", row=1, col=1)

    st.plotly_chart(fig_sens, use_container_width=True)

    # =========================================================================
    # Full Range Chart: Daily P&L from -5% to +2% spread (same format as survival chart)
    # =========================================================================
    st.markdown("---")
    st.markdown("**Full Spread Range: Daily P&L by Health Factor**")
    st.caption("Shows both downside (negative spread) and upside (positive spread). Bands show range from -5% to +2% spread.")

    # Spread bounds for full range analysis
    spread_worst_full = -0.05  # -5%
    spread_best_full = 0.02    # +2%

    # Reuse HF range and leverage calculations from earlier
    # aave_hf_valid, aave_lev_valid, twyne_hf_valid, twyne_lev_valid already defined

    # Compute daily P&L for each HF at both spread bounds
    # Daily P&L = (L * spread - cost) / 365
    aave_pnl_worst = (aave_lev_valid * spread_worst_full) * 100 / 365
    aave_pnl_best = (aave_lev_valid * spread_best_full) * 100 / 365
    twyne_pnl_worst = (twyne_lev_valid * spread_worst_full - twyne_lev_valid * twyne_psi * ir_u) * 100 / 365
    twyne_pnl_best = (twyne_lev_valid * spread_best_full - twyne_lev_valid * twyne_psi * ir_u) * 100 / 365

    fig_full_pnl = go.Figure()

    # Zero line
    fig_full_pnl.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # Aave band (orange)
    fig_full_pnl.add_trace(go.Scatter(x=aave_hf_valid, y=aave_pnl_best, mode='lines',
        line=dict(color='#ff9800', width=2.5),
        hovertemplate='Aave @ +2%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=aave_lev_valid, showlegend=False))
    fig_full_pnl.add_trace(go.Scatter(x=aave_hf_valid, y=aave_pnl_worst, mode='lines',
        line=dict(color='#ff9800', width=2.5, dash='dash'),
        hovertemplate='Aave @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=aave_lev_valid, fill='tonexty', fillcolor='rgba(255, 152, 0, 0.3)', showlegend=False))

    # Twyne band (blue)
    fig_full_pnl.add_trace(go.Scatter(x=twyne_hf_valid, y=twyne_pnl_best, mode='lines',
        line=dict(color='#2196f3', width=2),
        hovertemplate='Twyne @ +2%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=twyne_lev_valid, showlegend=False))
    fig_full_pnl.add_trace(go.Scatter(x=twyne_hf_valid, y=twyne_pnl_worst, mode='lines',
        line=dict(color='#2196f3', width=2, dash='dash'),
        hovertemplate='Twyne @ -5%<br>HF %{x:.2f} (L=%{customdata:.1f}x): %{y:.3f}%/day<extra></extra>',
        customdata=twyne_lev_valid, fill='tonexty', fillcolor='rgba(33, 150, 243, 0.2)', showlegend=False))

    # Legend traces
    fig_full_pnl.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Twyne', line=dict(color='#2196f3', width=3)))
    fig_full_pnl.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Aave', line=dict(color='#ff9800', width=3)))
    fig_full_pnl.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='-5% spread', line=dict(color='black', width=2, dash='dash')))
    fig_full_pnl.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='+2% spread', line=dict(color='black', width=2)))

    fig_full_pnl.update_layout(
        height=350, hovermode='x unified', margin=dict(l=60, t=80, r=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.18, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.9)')
    )
    fig_full_pnl.update_yaxes(title_text="Daily P&L % (up=gains, down=losses)")
    fig_full_pnl.update_xaxes(title_text="Health Factor", tickvals=key_hf, ticktext=[make_tick(hf) for hf in key_hf])

    st.plotly_chart(fig_full_pnl, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption("""
**Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results.
Always understand the risks before participating in DeFi.

Built with the Twyne stETH-ETH Looping Economics module.
""")
