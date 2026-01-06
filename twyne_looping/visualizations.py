"""
Visualization functions for Twyne stETH-ETH looping analysis.

All plotting functions support an `advanced` parameter:
    - advanced=False (default): Include explanatory annotations, color-coded
      risk zones, and educational labels for mixed audiences
    - advanced=True: Minimal annotations, just data and axis labels,
      suitable for DeFi-native users
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Optional, List, Tuple, Union
from pathlib import Path

from .economics import (
    psi, interest_rate, looped_yield, looped_yield_from_leverage,
    days_to_liquidation, leverage_from_ltv, ltv_from_leverage,
    ltv_from_hf, net_rate, compute_yield_grid, compute_days_to_liq_grid,
    DEFAULTS,
)


# =============================================================================
# Style Configuration
# =============================================================================

# Color scheme
COLORS = {
    "primary": "#2E86AB",      # Blue - main data
    "secondary": "#A23B72",    # Magenta - secondary data
    "positive": "#28A745",     # Green - positive/safe
    "negative": "#DC3545",     # Red - negative/danger
    "neutral": "#6C757D",      # Gray - neutral
    "highlight": "#F18F01",    # Orange - highlights
    "twyne": "#7B2CBF",        # Purple - Twyne branding
    "traditional": "#495057",  # Dark gray - traditional Aave
}

# Leverage level colors
LEVERAGE_COLORS = {
    5: COLORS["traditional"],
    10: "#4A90A4",
    20: COLORS["primary"],
    30: "#5E60CE",
    50: COLORS["twyne"],
}


def set_style(advanced: bool = False):
    """Set matplotlib style based on audience."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10 if advanced else 11,
        'axes.titlesize': 12 if advanced else 14,
        'axes.labelsize': 10 if advanced else 12,
        'figure.titlesize': 14 if advanced else 16,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# Historical Analysis Plots
# =============================================================================

def plot_historical_rates(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Axes:
    """
    Plot historical staking and borrowing rates over time.

    Args:
        df: DataFrame with r_stake and r_borrow columns (as decimals)
        ax: Matplotlib axes (created if None)
        advanced: If True, minimal annotations
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot rates
    ax.plot(df.index, df["r_stake"] * 100, label="stETH Yield (stake + lend)",
            color=COLORS["positive"], linewidth=1.2, alpha=0.8)
    ax.plot(df.index, df["r_borrow"] * 100, label="ETH Borrow Rate",
            color=COLORS["negative"], linewidth=1.2, alpha=0.8)

    # Fill between to show spread
    ax.fill_between(df.index, df["r_stake"] * 100, df["r_borrow"] * 100,
                    where=df["r_stake"] > df["r_borrow"],
                    alpha=0.2, color=COLORS["positive"], label="_nolegend_")
    ax.fill_between(df.index, df["r_stake"] * 100, df["r_borrow"] * 100,
                    where=df["r_stake"] <= df["r_borrow"],
                    alpha=0.2, color=COLORS["negative"], label="_nolegend_")

    ax.set_xlabel("Date")
    ax.set_ylabel("APR (%)")
    ax.legend(loc="upper right")

    if not advanced:
        ax.set_title("Historical stETH-ETH Rate Spread\n"
                     "Green = Profitable spread, Red = Negative spread")
        # Add annotation
        positive_pct = (df["r_stake"] > df["r_borrow"]).mean() * 100
        ax.annotate(f"Positive spread: {positive_pct:.1f}% of days",
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.set_title("stETH-ETH Historical Rates")

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    return ax


def plot_spread_distribution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Axes:
    """
    Plot histogram of daily spread values.

    Args:
        df: DataFrame with spread_raw column (as decimal)
        ax: Matplotlib axes (created if None)
        advanced: If True, minimal annotations
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    spread_pct = df["spread_raw"] * 100

    # Create histogram with color-coded bins
    n, bins, patches = ax.hist(spread_pct, bins=50, edgecolor='white', alpha=0.7)

    # Color bins based on positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge >= 0:
            patch.set_facecolor(COLORS["positive"])
        else:
            patch.set_facecolor(COLORS["negative"])

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=spread_pct.mean(), color=COLORS["primary"], linestyle='-',
               linewidth=2, label=f"Mean: {spread_pct.mean():.2f}%")
    ax.axvline(x=spread_pct.median(), color=COLORS["secondary"], linestyle=':',
               linewidth=2, label=f"Median: {spread_pct.median():.2f}%")

    ax.set_xlabel("Daily Spread (%)")
    ax.set_ylabel("Frequency (days)")
    ax.legend()

    if not advanced:
        ax.set_title("Distribution of Daily stETH-ETH Spread\n"
                     "Spread = stETH Yield - ETH Borrow Rate")
        positive_pct = (spread_pct > 0).mean() * 100
        ax.annotate(f"{positive_pct:.1f}% of days had\npositive spread",
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.set_title("Spread Distribution")

    return ax


def plot_rolling_spread(
    df: pd.DataFrame,
    windows: List[int] = [7, 30, 90],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Axes:
    """
    Plot rolling average of spread over time.

    Args:
        df: DataFrame with spread_raw column
        windows: Rolling window sizes
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = [COLORS["neutral"], COLORS["primary"], COLORS["twyne"]]

    for window, color in zip(windows, colors):
        col = f"spread_raw_ma{window}"
        if col in df.columns:
            ax.plot(df.index, df[col] * 100, label=f"{window}-day MA",
                    color=color, linewidth=1.5 if window > 7 else 1)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.fill_between(df.index, 0, df["spread_raw"] * 100,
                    where=df["spread_raw"] > 0,
                    alpha=0.1, color=COLORS["positive"])
    ax.fill_between(df.index, 0, df["spread_raw"] * 100,
                    where=df["spread_raw"] <= 0,
                    alpha=0.1, color=COLORS["negative"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (%)")
    ax.legend(loc="upper right")

    if not advanced:
        ax.set_title("Rolling Average Spread\n"
                     "Longer windows show persistent trends")
    else:
        ax.set_title("Rolling Spread")

    return ax


# =============================================================================
# Yield Comparison Plots (The Pitch)
# =============================================================================

def plot_yield_vs_leverage(
    r_stake: float,
    r_borrow: float,
    leverage_range: Tuple[float, float] = (1, 50),
    utilization: float = DEFAULTS["utilization"],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    **kwargs,
) -> plt.Axes:
    """
    Plot yield as a function of leverage.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        leverage_range: (min, max) leverage to plot
        utilization: CLP utilization
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size
        **kwargs: Override default parameters

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    params = {**DEFAULTS, **kwargs}

    # Calculate IR and Ψ
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    # Leverage range
    L = np.linspace(leverage_range[0], leverage_range[1], 200)
    liq_ltv_t = ltv_from_leverage(L) * 1.01  # Assume operating slightly below liq threshold

    psi_vals = psi(liq_ltv_t, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])
    yields = looped_yield_from_leverage(r_stake, r_borrow, L, ir_u, psi_vals)

    # Plot yield curve
    ax.plot(L, yields * 100, color=COLORS["twyne"], linewidth=2.5, label="Twyne Yield")

    # Mark key leverage points
    key_leverages = [5, 10, 20, 30, 50]
    for lev in key_leverages:
        if leverage_range[0] <= lev <= leverage_range[1]:
            idx = np.argmin(np.abs(L - lev))
            ax.scatter([L[idx]], [yields[idx] * 100], s=80,
                       color=LEVERAGE_COLORS.get(lev, COLORS["primary"]),
                       zorder=5, edgecolors='white', linewidth=1.5)

    # Traditional Aave limit (~5x)
    ax.axvline(x=5, color=COLORS["traditional"], linestyle='--',
               linewidth=1.5, alpha=0.7, label="Traditional Aave Max (~5x)")

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Leverage (x)")
    ax.set_ylabel("Net Yield (%)")
    ax.legend(loc="upper right" if yields[-1] > yields[0] else "lower right")

    if not advanced:
        raw_spread = (r_stake - r_borrow) * 100
        ax.set_title(f"Yield vs Leverage at Current Rates\n"
                     f"r_stake={r_stake*100:.2f}%, r_borrow={r_borrow*100:.2f}%, "
                     f"spread={raw_spread:.2f}%")

        # Add explanation
        ax.annotate("Higher leverage amplifies\nthe spread (positive or negative)",
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.set_title(f"Yield vs Leverage (spread={((r_stake-r_borrow)*100):.2f}%)")

    ax.set_xlim(leverage_range)

    return ax


def plot_leverage_comparison_timeseries(
    df: pd.DataFrame,
    leverage_levels: List[float] = [5, 10, 20, 50],
    utilization: float = DEFAULTS["utilization"],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (12, 6),
    **kwargs,
) -> plt.Axes:
    """
    Plot historical yield at different leverage levels.

    Args:
        df: DataFrame with r_stake and r_borrow columns
        leverage_levels: Leverage multiples to compare
        utilization: CLP utilization
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size
        **kwargs: Override default parameters

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    params = {**DEFAULTS, **kwargs}
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    for L in leverage_levels:
        liq_ltv_t = ltv_from_leverage(L) * 1.01
        psi_val = psi(liq_ltv_t, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])

        yields = looped_yield_from_leverage(df["r_stake"], df["r_borrow"], L, ir_u, psi_val)

        color = LEVERAGE_COLORS.get(L, COLORS["primary"])
        label = f"{L}x" + (" (Aave max)" if L == 5 else " (Twyne)" if L > 5 else "")
        ax.plot(df.index, yields * 100, label=label, color=color, linewidth=1.2, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Net Yield (%)")
    ax.legend(loc="upper right")

    if not advanced:
        ax.set_title("Historical Yield at Different Leverage Levels\n"
                     "Higher leverage = higher potential returns (and risks)")
    else:
        ax.set_title("Yield by Leverage Over Time")

    return ax


def plot_twyne_advantage_histogram(
    df: pd.DataFrame,
    twyne_leverage: float = 20,
    aave_leverage: float = 5,
    utilization: float = DEFAULTS["utilization"],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (10, 5),
    **kwargs,
) -> plt.Axes:
    """
    Plot histogram comparing Twyne vs traditional Aave yields.

    Args:
        df: DataFrame with r_stake and r_borrow columns
        twyne_leverage: Twyne leverage level
        aave_leverage: Traditional Aave leverage
        utilization: CLP utilization
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size
        **kwargs: Override default parameters

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    params = {**DEFAULTS, **kwargs}
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    # Compute yields for both leverage levels
    for L, name, color in [(aave_leverage, "Traditional Aave", COLORS["traditional"]),
                            (twyne_leverage, "Twyne", COLORS["twyne"])]:
        liq_ltv_t = ltv_from_leverage(L) * 1.01
        psi_val = psi(liq_ltv_t, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])
        yields = looped_yield_from_leverage(df["r_stake"], df["r_borrow"], L, ir_u, psi_val)

        label = f"{name} ({L}x): mean={yields.mean()*100:.1f}%"
        ax.hist(yields * 100, bins=50, alpha=0.6, label=label, color=color, edgecolor='white')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Daily Net Yield (%)")
    ax.set_ylabel("Frequency (days)")
    ax.legend()

    if not advanced:
        ax.set_title(f"Yield Distribution: Twyne {twyne_leverage}x vs Aave {aave_leverage}x\n"
                     "Twyne enables higher leverage = higher potential yields")
    else:
        ax.set_title(f"Yield Distribution: {twyne_leverage}x vs {aave_leverage}x")

    return ax


# =============================================================================
# Heatmap Plots (Reproducing Spreadsheet)
# =============================================================================

def plot_yield_heatmap(
    r_stake: float,
    r_borrow: float,
    hf_range: Tuple[float, float] = (1.0, 1.1),
    liq_ltv_range: Tuple[float, float] = (0.95, 0.98),
    n_hf: int = 20,
    n_ltv: int = 10,
    utilization: float = DEFAULTS["utilization"],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    **kwargs,
) -> plt.Axes:
    """
    Plot yield heatmap over HF and liquidation LTV grid.

    Reproduces the structure of the Detailed Sheet yield matrix.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        hf_range: (min, max) health factor range
        liq_ltv_range: (min, max) liquidation LTV range
        n_hf: Number of HF values
        n_ltv: Number of LTV values
        utilization: CLP utilization
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size
        **kwargs: Override default parameters

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create grid
    hf_values = np.linspace(hf_range[1], hf_range[0], n_hf)  # High to low (like spreadsheet)
    liq_ltv_values = np.linspace(liq_ltv_range[0], liq_ltv_range[1], n_ltv)

    # Compute yield grid
    yield_grid = compute_yield_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                     utilization=utilization, **kwargs)

    # Create diverging colormap centered at 0
    vmax = max(abs(yield_grid.min()), abs(yield_grid.max()))
    vmin = -vmax

    cmap = plt.cm.RdYlGn  # Red (negative) to Green (positive)
    norm = mcolors.TwoSlopeNorm(vmin=vmin * 100, vcenter=0, vmax=vmax * 100)

    im = ax.imshow(yield_grid * 100, cmap=cmap, norm=norm, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Net Yield (%)")

    # Set ticks
    ax.set_xticks(np.arange(n_ltv))
    ax.set_xticklabels([f"{v*100:.1f}%" for v in liq_ltv_values], rotation=45, ha='right')
    ax.set_yticks(np.arange(n_hf))
    ax.set_yticklabels([f"{v:.2f}" for v in hf_values])

    ax.set_xlabel("Boosted Liquidation LTV (λ̃_t)")
    ax.set_ylabel("Health Factor")

    if not advanced:
        ax.set_title(f"Net Yield by Health Factor & Liquidation LTV\n"
                     f"r_stake={r_stake*100:.2f}%, r_borrow={r_borrow*100:.2f}%")

        # Add annotation explaining axes
        ax.annotate("↓ Lower HF = Higher leverage\n→ Higher LTV = More Twyne boost",
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.set_title("Yield Heatmap (HF × Liq LTV)")

    return ax


def plot_days_to_liq_heatmap(
    r_stake: float,
    r_borrow: float,
    hf_range: Tuple[float, float] = (1.0, 1.1),
    liq_ltv_range: Tuple[float, float] = (0.95, 0.98),
    n_hf: int = 20,
    n_ltv: int = 10,
    utilization: float = DEFAULTS["utilization"],
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    max_days: float = 365,
    **kwargs,
) -> plt.Axes:
    """
    Plot days-to-liquidation heatmap.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        hf_range: (min, max) health factor range
        liq_ltv_range: (min, max) liquidation LTV range
        n_hf: Number of HF values
        n_ltv: Number of LTV values
        utilization: CLP utilization
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size
        max_days: Cap for display (inf values shown as this)
        **kwargs: Override default parameters

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create grid
    hf_values = np.linspace(hf_range[1], hf_range[0], n_hf)
    liq_ltv_values = np.linspace(liq_ltv_range[0], liq_ltv_range[1], n_ltv)

    # Compute days to liquidation grid
    dtl_grid = compute_days_to_liq_grid(r_stake, r_borrow, hf_values, liq_ltv_values,
                                         utilization=utilization, **kwargs)

    # Cap values for display
    dtl_display = np.clip(dtl_grid, 0, max_days)
    dtl_display[np.isinf(dtl_grid)] = max_days

    # Use reversed colormap (more days = safer = greener)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(dtl_display, cmap=cmap, aspect='auto', vmin=0, vmax=max_days)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Days to Liquidation")
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" if t < max_days else "SAFE" for t in ticks])

    # Set ticks
    ax.set_xticks(np.arange(n_ltv))
    ax.set_xticklabels([f"{v*100:.1f}%" for v in liq_ltv_values], rotation=45, ha='right')
    ax.set_yticks(np.arange(n_hf))
    ax.set_yticklabels([f"{v:.2f}" for v in hf_values])

    ax.set_xlabel("Boosted Liquidation LTV (λ̃_t)")
    ax.set_ylabel("Health Factor")

    if not advanced:
        ax.set_title(f"Days to Liquidation (if spread stays negative)\n"
                     f"r_stake={r_stake*100:.2f}%, r_borrow={r_borrow*100:.2f}%")
        ax.annotate("Green = SAFE (spread positive or many days to react)",
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.set_title("Days to Liquidation Heatmap")

    return ax


# =============================================================================
# Risk Transparency Plots
# =============================================================================

def plot_negative_spread_analysis(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    advanced: bool = False,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Axes:
    """
    Highlight periods with negative spread.

    Args:
        df: DataFrame with spread_raw column
        ax: Matplotlib axes
        advanced: If True, minimal annotations
        figsize: Figure size

    Returns:
        Matplotlib axes
    """
    set_style(advanced)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    spread_pct = df["spread_raw"] * 100

    # Plot spread
    ax.fill_between(df.index, 0, spread_pct,
                    where=spread_pct > 0,
                    alpha=0.5, color=COLORS["positive"], label="Positive spread")
    ax.fill_between(df.index, 0, spread_pct,
                    where=spread_pct <= 0,
                    alpha=0.5, color=COLORS["negative"], label="Negative spread")

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Find longest negative streak
    negative = df["spread_raw"] <= 0
    streaks = []
    current_streak = 0
    streak_start = None
    for i, (idx, neg) in enumerate(negative.items()):
        if neg:
            if current_streak == 0:
                streak_start = idx
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append((streak_start, current_streak))
            current_streak = 0
    if current_streak > 0:
        streaks.append((streak_start, current_streak))

    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (%)")
    ax.legend(loc="upper right")

    if not advanced:
        negative_pct = (df["spread_raw"] <= 0).mean() * 100
        max_streak = max([s[1] for s in streaks], default=0)
        ax.set_title(f"Negative Spread Periods: Risk Transparency\n"
                     f"{negative_pct:.1f}% of days negative, longest streak: {max_streak} days")
    else:
        ax.set_title("Spread Over Time")

    return ax


# =============================================================================
# Multi-Panel Summary Plots
# =============================================================================

def plot_summary_dashboard(
    df: pd.DataFrame,
    r_stake: Optional[float] = None,
    r_borrow: Optional[float] = None,
    utilization: float = DEFAULTS["utilization"],
    advanced: bool = False,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> plt.Figure:
    """
    Create a comprehensive summary dashboard.

    Args:
        df: DataFrame with rate data
        r_stake: Staking rate for heatmaps (default: use latest from df)
        r_borrow: Borrow rate for heatmaps (default: use latest from df)
        utilization: CLP utilization
        advanced: If True, minimal annotations
        figsize: Figure size
        save_path: Optional path to save figure
        **kwargs: Override default parameters

    Returns:
        Matplotlib figure
    """
    set_style(advanced)

    if r_stake is None:
        r_stake = df["r_stake"].iloc[-1]
    if r_borrow is None:
        r_borrow = df["r_borrow"].iloc[-1]

    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top left: Historical rates
    ax1 = fig.add_subplot(gs[0, 0])
    plot_historical_rates(df, ax=ax1, advanced=advanced)

    # Top middle: Spread distribution
    ax2 = fig.add_subplot(gs[0, 1])
    plot_spread_distribution(df, ax=ax2, advanced=advanced)

    # Top right: Yield vs Leverage
    ax3 = fig.add_subplot(gs[0, 2])
    plot_yield_vs_leverage(r_stake, r_borrow, ax=ax3, advanced=advanced,
                           utilization=utilization, **kwargs)

    # Bottom left: Leverage comparison time series
    ax4 = fig.add_subplot(gs[1, 0])
    plot_leverage_comparison_timeseries(df, ax=ax4, advanced=advanced,
                                        utilization=utilization, **kwargs)

    # Bottom middle: Yield heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    plot_yield_heatmap(r_stake, r_borrow, ax=ax5, advanced=advanced,
                       utilization=utilization, n_hf=10, n_ltv=6, **kwargs)

    # Bottom right: Days to liquidation heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    plot_days_to_liq_heatmap(r_stake, r_borrow, ax=ax6, advanced=advanced,
                             utilization=utilization, n_hf=10, n_ltv=6, **kwargs)

    if not advanced:
        fig.suptitle("Twyne stETH-ETH Looping Economics Dashboard", fontsize=16, y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
