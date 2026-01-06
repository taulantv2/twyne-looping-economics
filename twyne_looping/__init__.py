"""
Twyne stETH-ETH Looping Economics Module

This module provides tools for analyzing the economics of leveraged stETH-ETH
yield farming using the Twyne credit delegation protocol.

Modules:
    economics: Core financial formulas (yield, HF decay, days-to-liquidation)
    data: Data loading and preprocessing utilities
    visualizations: Plotting functions with advanced/simple mode toggle
    backtest: Historical backtest simulation with cumulative returns
    live_rates: Fetch live rates from Aave and Lido APIs
"""

from .economics import (
    psi,
    interest_rate,
    looped_yield,
    looped_yield_from_leverage,
    hf_decay_factor,
    days_to_liquidation,
    leverage_from_ltv,
    ltv_from_leverage,
    ltv_from_hf,
    net_rate,
    DEFAULTS,
)

from .data import (
    load_apr_data,
    compute_spreads,
    compute_rolling_stats,
    get_summary_stats,
    format_stats_report,
)

from .backtest import (
    BacktestResult,
    run_backtest,
    run_multi_leverage_backtest,
    compare_backtests,
    plot_backtest_results,
    plot_backtest_dashboard,
)

from .live_rates import (
    LiveRates,
    fetch_live_rates,
    fetch_lido_staking_rate,
    fetch_aave_rates,
    AVAILABLE_CHAINS,
)

__version__ = "0.1.0"
__all__ = [
    # Economics
    "psi",
    "interest_rate",
    "looped_yield",
    "looped_yield_from_leverage",
    "hf_decay_factor",
    "days_to_liquidation",
    "leverage_from_ltv",
    "ltv_from_leverage",
    "ltv_from_hf",
    "net_rate",
    "DEFAULTS",
    # Data
    "load_apr_data",
    "compute_spreads",
    "compute_rolling_stats",
    "get_summary_stats",
    "format_stats_report",
    # Backtest
    "BacktestResult",
    "run_backtest",
    "run_multi_leverage_backtest",
    "compare_backtests",
    "plot_backtest_results",
    "plot_backtest_dashboard",
    # Live Rates
    "LiveRates",
    "fetch_live_rates",
    "fetch_lido_staking_rate",
    "fetch_aave_rates",
    "AVAILABLE_CHAINS",
]
