"""
Backtesting module for Twyne stETH-ETH looping strategy.

Provides simulation of cumulative returns with health factor evolution,
liquidation events, and performance metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

from .economics import (
    psi, interest_rate, looped_yield_from_leverage, hf_decay_factor,
    leverage_from_ltv, ltv_from_leverage, DEFAULTS,
)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    # Time series data
    dates: pd.DatetimeIndex
    hf: np.ndarray           # Health factor over time
    pnl_daily: np.ndarray    # Daily P&L (as fraction of initial capital)
    pnl_cumulative: np.ndarray  # Cumulative P&L
    capital: np.ndarray      # Capital evolution (starting at 1.0)

    # Events
    liquidation_date: Optional[pd.Timestamp] = None
    liquidation_day: Optional[int] = None

    # Parameters used
    leverage: float = 0.0
    initial_hf: float = 0.0

    # Summary metrics
    metrics: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Compute summary metrics after initialization."""
        if len(self.pnl_cumulative) > 0:
            self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> Dict:
        """Compute summary performance metrics."""
        if self.liquidation_date is not None:
            # Position was liquidated
            total_return = -1.0  # Lost everything
            n_days = self.liquidation_day
        else:
            total_return = self.pnl_cumulative[-1]
            n_days = len(self.dates)

        # Annualized return (if not liquidated)
        if n_days > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (365 / n_days) - 1
        else:
            annualized_return = -1.0

        # Sharpe ratio (simplified, daily)
        if len(self.pnl_daily) > 1:
            daily_std = np.std(self.pnl_daily)
            if daily_std > 0:
                sharpe = np.mean(self.pnl_daily) / daily_std * np.sqrt(365)
            else:
                sharpe = np.nan
        else:
            sharpe = np.nan

        # Max drawdown
        cumulative = 1 + self.pnl_cumulative
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdown)

        # Min HF
        min_hf = np.min(self.hf[~np.isnan(self.hf)]) if len(self.hf) > 0 else np.nan

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "min_hf": min_hf,
            "n_days": n_days,
            "was_liquidated": self.liquidation_date is not None,
        }

    def summary(self) -> str:
        """Return formatted summary string."""
        m = self.metrics
        lines = [
            f"Backtest Results (Leverage: {self.leverage}x, Initial HF: {self.initial_hf:.2f})",
            "=" * 60,
            f"Duration: {m['n_days']} days",
            f"Total Return: {m['total_return']*100:.2f}%",
            f"Annualized Return: {m['annualized_return']*100:.2f}%",
            f"Sharpe Ratio: {m['sharpe_ratio']:.2f}" if not np.isnan(m['sharpe_ratio']) else "Sharpe Ratio: N/A",
            f"Max Drawdown: {m['max_drawdown']*100:.2f}%",
            f"Min Health Factor: {m['min_hf']:.3f}",
        ]
        if m['was_liquidated']:
            lines.append(f"LIQUIDATED on day {self.liquidation_day} ({self.liquidation_date.strftime('%Y-%m-%d')})")
        else:
            lines.append("Position remained solvent")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    leverage: float,
    initial_hf: float = 1.05,
    utilization: float = DEFAULTS["utilization"],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    **kwargs,
) -> BacktestResult:
    """
    Run a backtest simulation for a leveraged looping position.

    The simulation:
    1. Enters a position at the specified leverage and initial HF
    2. Each day, computes yield based on that day's rates
    3. Tracks HF evolution (decays/grows based on spread)
    4. If HF reaches 1.0, position is liquidated

    Args:
        df: DataFrame with r_stake and r_borrow columns (as decimals)
        leverage: Leverage multiple
        initial_hf: Starting health factor
        utilization: CLP utilization for IR calculation
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
        **kwargs: Override default parameters

    Returns:
        BacktestResult with full simulation data and metrics
    """
    params = {**DEFAULTS, **kwargs}

    # Filter date range
    data = df.copy()
    if start_date is not None:
        data = data[data.index >= start_date]
    if end_date is not None:
        data = data[data.index <= end_date]

    n_days = len(data)
    if n_days == 0:
        raise ValueError("No data in specified date range")

    # Calculate fixed parameters
    liq_ltv_t = ltv_from_leverage(leverage) * initial_hf  # λ̃_t such that at initial_hf we're at target leverage
    psi_val = psi(liq_ltv_t, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    # Initialize arrays
    hf = np.zeros(n_days)
    pnl_daily = np.zeros(n_days)
    pnl_cumulative = np.zeros(n_days)
    capital = np.ones(n_days)

    hf[0] = initial_hf
    liquidation_date = None
    liquidation_day = None

    # Simulate day by day
    for i, (date, row) in enumerate(data.iterrows()):
        r_stake = row["r_stake"]
        r_borrow = row["r_borrow"]

        if i > 0:
            # Update HF based on previous day's rates
            decay = hf_decay_factor(r_stake, r_borrow, ir_u, psi_val)
            # Convert annual factor to daily
            daily_decay = decay ** (1/365)
            hf[i] = hf[i-1] * daily_decay

            # Check for liquidation
            if hf[i] <= 1.0:
                hf[i] = 1.0
                liquidation_date = date
                liquidation_day = i
                # Mark remaining days as NaN
                hf[i+1:] = np.nan
                pnl_daily[i:] = np.nan
                pnl_cumulative[i:] = -1.0  # Lost everything
                capital[i:] = 0.0
                break

        # Calculate daily yield at current effective leverage
        # Effective leverage changes as HF changes: L_eff = 1/(1 - λ̃_t/HF)
        effective_ltv = liq_ltv_t / hf[i]
        effective_leverage = leverage_from_ltv(effective_ltv)

        daily_yield = looped_yield_from_leverage(
            r_stake, r_borrow, effective_leverage, ir_u, psi_val
        ) / 365  # Convert annual to daily

        pnl_daily[i] = daily_yield

        if i == 0:
            pnl_cumulative[i] = daily_yield
            capital[i] = 1.0 + daily_yield
        else:
            pnl_cumulative[i] = pnl_cumulative[i-1] + daily_yield * capital[i-1]
            capital[i] = capital[i-1] * (1 + daily_yield)

    return BacktestResult(
        dates=data.index,
        hf=hf,
        pnl_daily=pnl_daily,
        pnl_cumulative=pnl_cumulative,
        capital=capital,
        liquidation_date=liquidation_date,
        liquidation_day=liquidation_day,
        leverage=leverage,
        initial_hf=initial_hf,
    )


def run_multi_leverage_backtest(
    df: pd.DataFrame,
    leverage_levels: List[float] = [5, 10, 20, 30, 50],
    initial_hf: float = 1.05,
    utilization: float = DEFAULTS["utilization"],
    **kwargs,
) -> Dict[float, BacktestResult]:
    """
    Run backtests for multiple leverage levels.

    Args:
        df: DataFrame with rate data
        leverage_levels: List of leverage multiples to test
        initial_hf: Starting health factor
        utilization: CLP utilization
        **kwargs: Override default parameters

    Returns:
        Dictionary mapping leverage to BacktestResult
    """
    results = {}
    for L in leverage_levels:
        results[L] = run_backtest(df, L, initial_hf, utilization, **kwargs)
    return results


def compare_backtests(results: Dict[float, BacktestResult]) -> pd.DataFrame:
    """
    Create comparison DataFrame from multiple backtest results.

    Args:
        results: Dictionary mapping leverage to BacktestResult

    Returns:
        DataFrame with metrics for each leverage level
    """
    rows = []
    for L, result in sorted(results.items()):
        row = {"leverage": L, **result.metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("leverage")

    # Format for display
    df["total_return"] = df["total_return"] * 100
    df["annualized_return"] = df["annualized_return"] * 100
    df["max_drawdown"] = df["max_drawdown"] * 100

    return df


# =============================================================================
# Visualization helpers for backtest results
# =============================================================================

def plot_backtest_results(
    results: Union[BacktestResult, Dict[float, BacktestResult]],
    ax=None,
    figsize: Tuple[int, int] = (12, 5),
    plot_type: str = "capital",  # "capital", "hf", or "pnl"
    advanced: bool = False,
):
    """
    Plot backtest results.

    Args:
        results: Single BacktestResult or dict of results
        ax: Matplotlib axes
        figsize: Figure size
        plot_type: What to plot ("capital", "hf", or "pnl")
        advanced: If True, minimal annotations

    Returns:
        Matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Handle single result vs multiple
    if isinstance(results, BacktestResult):
        results = {results.leverage: results}

    colors = {
        5: "#495057",
        10: "#4A90A4",
        20: "#2E86AB",
        30: "#5E60CE",
        50: "#7B2CBF",
    }

    for L, result in sorted(results.items()):
        color = colors.get(L, "#2E86AB")
        label = f"{L}x"

        if result.liquidation_date is not None:
            label += f" (liq day {result.liquidation_day})"

        if plot_type == "capital":
            y = result.capital
            ylabel = "Capital (starting at 1.0)"
        elif plot_type == "hf":
            y = result.hf
            ylabel = "Health Factor"
        elif plot_type == "pnl":
            y = result.pnl_cumulative * 100
            ylabel = "Cumulative P&L (%)"
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        ax.plot(result.dates, y, label=label, color=color, linewidth=1.5)

        # Mark liquidation point
        if result.liquidation_date is not None and plot_type != "pnl":
            ax.axvline(x=result.liquidation_date, color=color, linestyle='--', alpha=0.5)

    if plot_type == "hf":
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label="Liquidation threshold")

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right" if plot_type == "capital" else "best")

    title_map = {
        "capital": "Capital Evolution",
        "hf": "Health Factor Evolution",
        "pnl": "Cumulative P&L",
    }

    if not advanced:
        ax.set_title(f"Backtest: {title_map[plot_type]}\n"
                     "Higher leverage = higher returns but higher liquidation risk")
    else:
        ax.set_title(title_map[plot_type])

    return ax


def plot_backtest_dashboard(
    results: Dict[float, BacktestResult],
    figsize: Tuple[int, int] = (14, 8),
    advanced: bool = False,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Create comprehensive backtest dashboard.

    Args:
        results: Dictionary mapping leverage to BacktestResult
        figsize: Figure size
        advanced: If True, minimal annotations
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Capital evolution
    plot_backtest_results(results, ax=axes[0, 0], plot_type="capital", advanced=advanced)

    # Health factor evolution
    plot_backtest_results(results, ax=axes[0, 1], plot_type="hf", advanced=advanced)

    # Cumulative P&L
    plot_backtest_results(results, ax=axes[1, 0], plot_type="pnl", advanced=advanced)

    # Metrics table
    ax_table = axes[1, 1]
    ax_table.axis('off')

    comparison_df = compare_backtests(results)
    table_data = []
    headers = ["Leverage", "Total\nReturn", "Annual\nReturn", "Max\nDrawdown", "Liquidated"]

    for L in sorted(results.keys()):
        row = comparison_df.loc[L]
        table_data.append([
            f"{L}x",
            f"{row['total_return']:.1f}%",
            f"{row['annualized_return']:.1f}%",
            f"{row['max_drawdown']:.1f}%",
            "Yes" if row['was_liquidated'] else "No",
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    if not advanced:
        fig.suptitle("Twyne Looping Backtest Results\n"
                     "Historical simulation with different leverage levels", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
