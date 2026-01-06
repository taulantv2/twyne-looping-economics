"""
Data loading and preprocessing for Twyne stETH-ETH looping analysis.

Handles loading historical APR data from CSV and computing derived quantities
like spreads and rolling statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List


def load_apr_data(
    csv_path: Union[str, Path],
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load historical APR data from CSV.

    Expected columns:
        - datetime: Date of observation
        - ETH_Borrow: Aave ETH borrow rate (r_borrow)
        - stETH_Supply: stETH lending rate on Aave
        - LIDO_stETH_Staking: Lido staking yield
        - stETH_staking_and_lending: Combined yield (r_stake)

    Args:
        csv_path: Path to the CSV file
        parse_dates: Whether to parse datetime column

    Returns:
        DataFrame with cleaned data, rates as decimals (not percentages)
    """
    df = pd.read_csv(csv_path)

    # Parse datetime
    if parse_dates and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    # Convert percentage strings to decimals
    rate_columns = ["ETH_Borrow", "stETH_Supply", "LIDO_stETH_Staking", "stETH_staking_and_lending"]
    for col in rate_columns:
        if col in df.columns:
            if df[col].dtype == object:
                # Remove '%' and convert
                df[col] = df[col].str.rstrip('%').astype(float) / 100.0
            elif df[col].max() > 1:
                # Already numeric but in percentage form
                df[col] = df[col] / 100.0

    # Rename for clarity
    df = df.rename(columns={
        "ETH_Borrow": "r_borrow",
        "stETH_Supply": "r_steth_supply",
        "LIDO_stETH_Staking": "r_lido_staking",
        "stETH_staking_and_lending": "r_stake",
    })

    return df


def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread columns for analysis.

    Adds columns:
        - spread_raw: r_stake - r_borrow (unlevered spread)
        - spread_positive: Boolean, True if spread > 0

    Args:
        df: DataFrame with r_stake and r_borrow columns

    Returns:
        DataFrame with additional spread columns
    """
    df = df.copy()

    # Raw spread (unlevered)
    df["spread_raw"] = df["r_stake"] - df["r_borrow"]

    # Boolean indicator for positive spread days
    df["spread_positive"] = df["spread_raw"] > 0

    return df


def compute_rolling_stats(
    df: pd.DataFrame,
    windows: List[int] = [7, 30, 90],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute rolling statistics for specified columns.

    Args:
        df: Input DataFrame
        windows: List of window sizes (in days)
        columns: Columns to compute stats for (default: spread_raw, r_stake, r_borrow)

    Returns:
        DataFrame with additional rolling mean columns
    """
    df = df.copy()

    if columns is None:
        columns = ["spread_raw", "r_stake", "r_borrow"]

    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            df[f"{col}_ma{window}"] = df[col].rolling(window=window, min_periods=1).mean()

    return df


def compute_yield_series(
    df: pd.DataFrame,
    leverage: float,
    ir_u: float,
    psi_val: float,
) -> pd.Series:
    """
    Compute daily yield at a fixed leverage level.

    Y = L · r_stake - (L-1) · r_borrow - L · IR(u) · Ψ

    Args:
        df: DataFrame with r_stake and r_borrow columns
        leverage: Leverage multiple
        ir_u: Twyne interest rate at utilization
        psi_val: CLP cost factor Ψ

    Returns:
        Series of daily yields
    """
    return (
        leverage * df["r_stake"]
        - (leverage - 1) * df["r_borrow"]
        - leverage * ir_u * psi_val
    )


def compute_yield_comparison(
    df: pd.DataFrame,
    leverage_levels: List[float],
    ir_u: float,
    psi_val: float,
) -> pd.DataFrame:
    """
    Compute daily yields for multiple leverage levels.

    Args:
        df: DataFrame with r_stake and r_borrow columns
        leverage_levels: List of leverage multiples to compute
        ir_u: Twyne interest rate at utilization
        psi_val: CLP cost factor Ψ

    Returns:
        DataFrame with yield columns for each leverage level
    """
    result = df[["r_stake", "r_borrow"]].copy()

    for L in leverage_levels:
        col_name = f"yield_{L:.0f}x" if L == int(L) else f"yield_{L:.1f}x"
        result[col_name] = compute_yield_series(df, L, ir_u, psi_val)

    return result


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the dataset.

    Args:
        df: DataFrame with rate and spread columns

    Returns:
        Dictionary of summary statistics
    """
    stats = {
        "date_range": {
            "start": df.index.min().strftime("%Y-%m-%d") if hasattr(df.index, 'min') else None,
            "end": df.index.max().strftime("%Y-%m-%d") if hasattr(df.index, 'max') else None,
            "n_days": len(df),
        },
        "r_stake": {
            "mean": df["r_stake"].mean(),
            "std": df["r_stake"].std(),
            "min": df["r_stake"].min(),
            "max": df["r_stake"].max(),
            "median": df["r_stake"].median(),
        },
        "r_borrow": {
            "mean": df["r_borrow"].mean(),
            "std": df["r_borrow"].std(),
            "min": df["r_borrow"].min(),
            "max": df["r_borrow"].max(),
            "median": df["r_borrow"].median(),
        },
    }

    if "spread_raw" in df.columns:
        stats["spread"] = {
            "mean": df["spread_raw"].mean(),
            "std": df["spread_raw"].std(),
            "min": df["spread_raw"].min(),
            "max": df["spread_raw"].max(),
            "median": df["spread_raw"].median(),
            "pct_positive": (df["spread_raw"] > 0).mean() * 100,
        }

    return stats


def format_stats_report(stats: dict) -> str:
    """
    Format summary statistics as a readable report.

    Args:
        stats: Dictionary from get_summary_stats()

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "Historical APR Data Summary",
        "=" * 60,
        f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}",
        f"Total Days: {stats['date_range']['n_days']}",
        "",
        "Staking + Lending Rate (r_stake):",
        f"  Mean:   {stats['r_stake']['mean']*100:6.2f}%",
        f"  Median: {stats['r_stake']['median']*100:6.2f}%",
        f"  Std:    {stats['r_stake']['std']*100:6.2f}%",
        f"  Range:  {stats['r_stake']['min']*100:6.2f}% to {stats['r_stake']['max']*100:6.2f}%",
        "",
        "ETH Borrow Rate (r_borrow):",
        f"  Mean:   {stats['r_borrow']['mean']*100:6.2f}%",
        f"  Median: {stats['r_borrow']['median']*100:6.2f}%",
        f"  Std:    {stats['r_borrow']['std']*100:6.2f}%",
        f"  Range:  {stats['r_borrow']['min']*100:6.2f}% to {stats['r_borrow']['max']*100:6.2f}%",
    ]

    if "spread" in stats:
        lines.extend([
            "",
            "Raw Spread (r_stake - r_borrow):",
            f"  Mean:   {stats['spread']['mean']*100:6.2f}%",
            f"  Median: {stats['spread']['median']*100:6.2f}%",
            f"  Std:    {stats['spread']['std']*100:6.2f}%",
            f"  Range:  {stats['spread']['min']*100:6.2f}% to {stats['spread']['max']*100:6.2f}%",
            f"  Days Positive: {stats['spread']['pct_positive']:.1f}%",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)
