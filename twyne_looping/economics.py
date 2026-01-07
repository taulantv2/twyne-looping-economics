"""
Core economic formulas for Twyne stETH-ETH looping.

This module implements the mathematical relationships from the Twyne V1 Whitepaper
for calculating yields, health factor dynamics, and liquidation timelines.

Notation (matching whitepaper):
    λ_t     : Operating LTV (debt/collateral ratio)
    λ̃_t    : Boosted Liquidation LTV (Twyne liquidation threshold)
    λ̃_e^C  : Liquidation LTV of borrower's collateral on external market (Aave)
    λ̃_e^CLP: Liquidation LTV of CLP collateral on external market (Aave)
    β_safe  : Safety buffer parameter
    Ψ       : CLP cost factor (ratio of CLP collateral to borrower collateral)
    HF      : Health Factor = λ̃_t / λ_t
    L       : Leverage = 1 / (1 - λ_t)
"""

import numpy as np
from typing import Union, Optional

# Type alias for numeric inputs (scalar or array)
Numeric = Union[float, np.ndarray]


# =============================================================================
# Default Parameters (from Detailed Sheet)
# =============================================================================

DEFAULTS = {
    # External market parameters (Aave)
    "liq_ltv_c": 0.95,       # λ̃_e^C: Liquidation LTV for borrower collateral (stETH)
    "liq_ltv_clp": 0.95,     # λ̃_e^CLP: Liquidation LTV for CLP collateral (stETH)

    # Twyne parameters
    "beta_safe": 1.0,        # Safety buffer
    "liq_ltv_t_max": 0.98,   # λ̃_t^max: Maximum boosted liquidation LTV

    # Interest rate model parameters (Whitepaper Section 5)
    "I_min": 0.00,           # Minimum interest rate
    "I_0": 0.008,            # Interest rate at kink (0.8%)
    "u_0": 0.90,             # Utilization at kink
    "I_max": 0.20,           # Maximum interest rate (20%)
    "gamma": 32,             # Curvature parameter

    # Default utilization for calculations
    "utilization": 0.95,
}


# =============================================================================
# Core Formulas
# =============================================================================

def psi(
    liq_ltv_t: Numeric,
    beta_safe: float = DEFAULTS["beta_safe"],
    liq_ltv_clp: float = DEFAULTS["liq_ltv_clp"],
    liq_ltv_c: float = DEFAULTS["liq_ltv_c"],
) -> Numeric:
    """
    Calculate the CLP cost factor Ψ.

    This represents the ratio of CLP collateral to borrower collateral required
    to support the boosted position. Derived from the credit reservation
    invariant (Whitepaper Equation 7).

    Ψ = λ̃_t / (β_safe · λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP

    Args:
        liq_ltv_t: Boosted liquidation LTV (λ̃_t)
        beta_safe: Safety buffer parameter
        liq_ltv_clp: Liquidation LTV of CLP collateral on Aave
        liq_ltv_c: Liquidation LTV of borrower's collateral on Aave

    Returns:
        CLP cost factor Ψ
    """
    return liq_ltv_t / (beta_safe * liq_ltv_clp) - liq_ltv_c / liq_ltv_clp


def interest_rate(
    u: Numeric,
    I_min: float = DEFAULTS["I_min"],
    I_0: float = DEFAULTS["I_0"],
    u_0: float = DEFAULTS["u_0"],
    I_max: float = DEFAULTS["I_max"],
    gamma: float = DEFAULTS["gamma"],
) -> Numeric:
    """
    Calculate the Twyne interest rate IR(u) as a function of utilization.

    From Whitepaper Section 5:
    IR(u) = I_min + (I_0 - I_min)/u_0 · u + (I_max - I_min - (I_0 - I_min)/u_0) · u^γ

    Args:
        u: Utilization (0 to 1)
        I_min: Minimum interest rate
        I_0: Interest rate at kink
        u_0: Utilization at kink
        I_max: Maximum interest rate
        gamma: Curvature parameter

    Returns:
        Interest rate IR(u)
    """
    linear_slope = (I_0 - I_min) / u_0
    nonlinear_coeff = I_max - I_min - linear_slope
    return I_min + linear_slope * u + nonlinear_coeff * np.power(u, gamma)


def leverage_from_ltv(lambda_t: Numeric) -> Numeric:
    """
    Calculate leverage from operating LTV.

    L = 1 / (1 - λ_t)

    Args:
        lambda_t: Operating LTV (0 to <1)

    Returns:
        Leverage multiple
    """
    return 1.0 / (1.0 - lambda_t)


def ltv_from_leverage(L: Numeric) -> Numeric:
    """
    Calculate operating LTV from leverage.

    λ_t = (L - 1) / L = 1 - 1/L

    Args:
        L: Leverage multiple (>1)

    Returns:
        Operating LTV
    """
    return 1.0 - 1.0 / L


def ltv_from_hf(liq_ltv_t: Numeric, hf: Numeric) -> Numeric:
    """
    Calculate operating LTV from health factor and liquidation LTV.

    λ_t = λ̃_t / HF

    Args:
        liq_ltv_t: Boosted liquidation LTV (λ̃_t)
        hf: Health factor

    Returns:
        Operating LTV (λ_t)
    """
    return liq_ltv_t / hf


def net_rate(
    r_stake: Numeric,
    r_borrow: Numeric,
    ir_u: Numeric,
    psi_val: Numeric,
) -> Numeric:
    """
    Calculate the net rate (determines HF decay direction).

    r_net = r_stake - r_borrow - Ψ · IR(u)

    When r_net > 0: HF improves over time (position is safe)
    When r_net < 0: HF decays over time (eventual liquidation)

    Args:
        r_stake: Staking + lending rate (stETH yield)
        r_borrow: Borrowing rate (ETH borrow cost)
        ir_u: Twyne interest rate at current utilization
        psi_val: CLP cost factor Ψ

    Returns:
        Net rate
    """
    return r_stake - r_borrow - psi_val * ir_u


def looped_yield(
    r_stake: Numeric,
    r_borrow: Numeric,
    lambda_t: Numeric,
    ir_u: Numeric,
    psi_val: Numeric,
) -> Numeric:
    """
    Calculate the looped net farming yield.

    Y = (r_stake - r_borrow · λ_t - IR(u) · Ψ) / (1 - λ_t)

    This is the annualized yield on initial capital for a leveraged position.

    Args:
        r_stake: Staking + lending rate (stETH yield)
        r_borrow: Borrowing rate (ETH borrow cost)
        lambda_t: Operating LTV
        ir_u: Twyne interest rate at current utilization
        psi_val: CLP cost factor Ψ

    Returns:
        Net yield (annualized)
    """
    numerator = r_stake - r_borrow * lambda_t - ir_u * psi_val
    denominator = 1.0 - lambda_t
    return numerator / denominator


def looped_yield_from_leverage(
    r_stake: Numeric,
    r_borrow: Numeric,
    leverage: Numeric,
    ir_u: Numeric,
    psi_val: Numeric,
) -> Numeric:
    """
    Calculate the looped net farming yield using leverage directly.

    Y = L · r_stake - (L-1) · r_borrow - L · IR(u) · Ψ

    Equivalent to looped_yield() but parameterized by leverage.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        leverage: Leverage multiple
        ir_u: Twyne interest rate
        psi_val: CLP cost factor Ψ

    Returns:
        Net yield (annualized)
    """
    return leverage * r_stake - (leverage - 1) * r_borrow - leverage * ir_u * psi_val


def hf_decay_factor(
    r_stake: Numeric,
    r_borrow: Numeric,
    ir_u: Numeric,
    psi_val: Numeric,
) -> Numeric:
    """
    Calculate the daily health factor decay/growth factor.

    HF' = HF · (1 + r_net / (1 + r_borrow))

    This function returns the multiplicative factor:
    factor = 1 + r_net / (1 + r_borrow)

    Where:
    - Collateral grows at: (1 + r_stake - Ψ·IR(u))
    - Debt grows at: (1 + r_borrow)
    - r_net = r_stake - r_borrow - Ψ·IR(u)

    Args:
        r_stake: Staking + lending rate (annualized)
        r_borrow: Borrowing rate (annualized)
        ir_u: Twyne interest rate
        psi_val: CLP cost factor Ψ

    Returns:
        Multiplicative factor for HF evolution (per year)
    """
    r_net = net_rate(r_stake, r_borrow, ir_u, psi_val)
    return 1.0 + r_net / (1.0 + r_borrow)


def days_to_liquidation(
    hf_0: Numeric,
    r_stake: Numeric,
    r_borrow: Numeric,
    ir_u: Numeric,
    psi_val: Numeric,
    days_per_year: float = 365.0,
) -> Numeric:
    """
    Calculate days until liquidation (HF reaches 1.0).

    T_liq = -365 · ln(HF_0) / ln(1 + r_net/(1 + r_borrow))

    Args:
        hf_0: Initial health factor
        r_stake: Staking + lending rate (annualized)
        r_borrow: Borrowing rate (annualized)
        ir_u: Twyne interest rate
        psi_val: CLP cost factor Ψ
        days_per_year: Days in a year (default 365)

    Returns:
        Days to liquidation. Returns np.inf if position is safe (r_net >= 0).
        Returns np.nan for invalid inputs (hf_0 <= 1).
    """
    r_net = net_rate(r_stake, r_borrow, ir_u, psi_val)
    decay_factor = 1.0 + r_net / (1.0 + r_borrow)

    # Ensure all inputs are arrays for consistent handling
    hf_0 = np.atleast_1d(np.asarray(hf_0, dtype=float))
    r_net = np.atleast_1d(np.asarray(r_net, dtype=float))
    decay_factor = np.atleast_1d(np.asarray(decay_factor, dtype=float))

    # Broadcast all arrays to same shape
    hf_0, r_net, decay_factor = np.broadcast_arrays(hf_0, r_net, decay_factor)

    result = np.full_like(hf_0, np.nan, dtype=float)

    # Valid positions: HF > 1, negative spread, positive decay factor
    valid = (hf_0 > 1.0) & (r_net < 0) & (decay_factor > 0)
    if np.any(valid):
        result[valid] = -days_per_year * np.log(hf_0[valid]) / np.log(decay_factor[valid])

    # Safe positions: HF > 1, non-negative spread
    safe = (hf_0 > 1.0) & (r_net >= 0)
    result[safe] = np.inf

    # Return scalar if input was scalar
    if result.size == 1:
        return float(result.flat[0])
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_yield_grid(
    r_stake: float,
    r_borrow: float,
    hf_values: np.ndarray,
    liq_ltv_values: np.ndarray,
    utilization: float = DEFAULTS["utilization"],
    **kwargs,
) -> np.ndarray:
    """
    Compute a yield grid over HF and liquidation LTV values.

    Reproduces the structure of the Detailed Sheet yield matrix.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        hf_values: Array of health factor values (rows)
        liq_ltv_values: Array of boosted liquidation LTV values (columns)
        utilization: CLP utilization for IR calculation
        **kwargs: Override default parameters

    Returns:
        2D array of yields, shape (len(hf_values), len(liq_ltv_values))
    """
    params = {**DEFAULTS, **kwargs}
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    # Create meshgrid
    HF, LIQ_LTV = np.meshgrid(hf_values, liq_ltv_values, indexing='ij')

    # Compute operating LTV
    lambda_t = ltv_from_hf(LIQ_LTV, HF)

    # Compute Ψ for each liquidation LTV
    psi_vals = psi(LIQ_LTV, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])

    # Compute yield
    return looped_yield(r_stake, r_borrow, lambda_t, ir_u, psi_vals)


def compute_days_to_liq_grid(
    r_stake: float,
    r_borrow: float,
    hf_values: np.ndarray,
    liq_ltv_values: np.ndarray,
    utilization: float = DEFAULTS["utilization"],
    **kwargs,
) -> np.ndarray:
    """
    Compute a days-to-liquidation grid over HF and liquidation LTV values.

    Args:
        r_stake: Staking + lending rate
        r_borrow: Borrowing rate
        hf_values: Array of health factor values (rows)
        liq_ltv_values: Array of boosted liquidation LTV values (columns)
        utilization: CLP utilization for IR calculation
        **kwargs: Override default parameters

    Returns:
        2D array of days to liquidation, shape (len(hf_values), len(liq_ltv_values))
    """
    params = {**DEFAULTS, **kwargs}
    ir_u = interest_rate(utilization, **{k: params[k] for k in ["I_min", "I_0", "u_0", "I_max", "gamma"]})

    # Create meshgrid
    HF, LIQ_LTV = np.meshgrid(hf_values, liq_ltv_values, indexing='ij')

    # Compute Ψ for each liquidation LTV
    psi_vals = psi(LIQ_LTV, params["beta_safe"], params["liq_ltv_clp"], params["liq_ltv_c"])

    # Compute days to liquidation
    return days_to_liquidation(HF, r_stake, r_borrow, ir_u, psi_vals)
