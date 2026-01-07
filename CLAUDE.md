# Twyne Boosted Looping Economics - Claude Context

## Project Overview

This project analyzes the economics of yield farming stETH-ETH interest rate spreads using the Twyne protocol for boosted leverage positions. It includes:
- **Interactive Streamlit dashboard** (`app.py`) for exploring economics
- **Python module** (`twyne_looping/`) with core financial formulas
- **Historical APR data** (2023-2025) for backtesting
- **Documentation** including whitepapers and technical notes

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (uses lockfile for reproducible builds)
uv sync

# Run the dashboard
uv run streamlit run app.py
```

---

## Key Documents

### 1. Twyne V1 Whitepaper (`docs/Twyne_V1_Whitepaper.pdf`)
- 32-page comprehensive whitepaper covering protocol mechanics
- Key concepts: Credit delegation, dual LTV framework, liquidation mechanics, interest rate model

### 2. Risk Modeling Thread (`docs/Twyne_thread.pdf`)
- Analysis of 1115 Euler V2 liquidation events (Sept 2024 - May 2025)
- Key finding: At 94% max LLTV, annualized CLP loss rate ~0.16%
- Profitability threshold: Just 0.3% utilization for CLPs to break even

### 3. Historical APR Data (`data/stETH-ETH_historical_APR_data.csv`)
- 987 rows of daily APR data (Jan 2023 - Oct 2025)
- Columns: datetime, ETH_Borrow, stETH_Supply, LIDO_stETH_Staking, stETH_staking_and_lending
- ETH borrow rates: ~2-4%
- Lido stETH staking rates: ~3-6%
- Spread typically positive (stETH staking > ETH borrowing)

### 4. Detailed Economics Spreadsheet (`docs/Boosted_Looping_Economics.ods`)
- Key sheet: "Detailed Sheet" (formulas documented in LaTeX technical notes)
- Other sheets: Twyne Looping, Historical Daily APRs, etc.

### 5. Technical Documentation (`docs/technical-notes/`)
- `detailed_sheet_technical_doc.tex` - LaTeX source with full formula derivations
- `detailed_sheet_technical_doc.pdf` - Compiled documentation

---

## Python Module (`twyne_looping/`)

### Core Modules

#### `economics.py` - Financial Formulas
```python
from twyne_looping import (
    psi,                    # CLP cost factor Ψ
    interest_rate,          # IR(u) from interest rate model
    looped_yield,           # Net yield Y at given parameters
    looped_yield_from_leverage,  # Y from leverage L
    hf_decay_factor,        # Health factor evolution rate
    days_to_liquidation,    # T_liq calculation
    leverage_from_ltv,      # L = 1/(1-λ_t)
    ltv_from_leverage,      # λ_t = (L-1)/L
    ltv_from_hf,           # λ_t = λ̃_t/HF
    net_rate,              # r_net = r_stake - r_borrow - Ψ·IR(u)
    DEFAULTS,              # Default parameter values
)
```

#### `data.py` - Data Loading & Preprocessing
```python
from twyne_looping import (
    load_apr_data,         # Load CSV data
    compute_spreads,       # Add spread columns
    compute_rolling_stats, # Rolling means (7, 30, 90-day)
    get_summary_stats,     # Statistical summary
    format_stats_report,   # Human-readable report
)
```

#### `backtest.py` - Backtesting Engine
```python
from twyne_looping import (
    BacktestResult,        # Dataclass with results
    run_backtest,          # Single leverage backtest
    run_multi_leverage_backtest,  # Multiple leverages
    compare_backtests,     # Comparative analysis
    plot_backtest_results, # Visualization
    plot_backtest_dashboard,  # Multi-panel view
)
```

#### `live_rates.py` - API Integration
```python
from twyne_looping import (
    LiveRates,             # Rate data container
    fetch_live_rates,      # Get current rates
    fetch_lido_staking_rate,  # Lido API
    fetch_aave_rates,      # Aave subgraph
    AVAILABLE_CHAINS,      # Supported chains
)
```

---

## Core Twyne Protocol Concepts

### Dual LTV Framework (CRITICAL - Notation Matters!)

**Tilde (λ̃) ALWAYS denotes LIQUIDATION LTV, not operating LTV.**

| Symbol | Name | Definition |
|--------|------|------------|
| λ_t | Operating LTV | B/C - current debt over borrower collateral |
| λ̃_t | Boosted Liquidation LTV | Liquidation threshold for Twyne position |
| λ̃_t^max | Max Boosted Liquidation LTV | Maximum allowable λ̃_t (e.g., 98%) |
| λ̃_e^C | Borrower Collateral Liq. LTV | Aave liquidation LTV for borrower's collateral (C) |
| λ̃_e^CLP | CLP Collateral Liq. LTV | Aave liquidation LTV for CLP collateral |

**Key Relationships:**
- **Leverage**: L = 1/(1 - λ_t), equivalently λ_t = (L-1)/L
- **Health Factor**: HF = λ̃_t / λ_t
- **At HF = 1**: λ_t = λ̃_t (operating at liquidation threshold)

### Credit Reservation Invariant (Whitepaper Eq. 7)

The CLP collateral required to support a borrower position:
```
C_LP = C · (λ̃_t / (β_safe · λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP)
```

This defines the **CLP cost factor Ψ**:
```
Ψ = λ̃_t / (β_safe · λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP
```

Where:
- C = Borrower collateral
- β_safe = Safety buffer (typically 0.85-1.0)
- Ψ = Ratio of CLP collateral to borrower collateral

### Interest Rate Model (Whitepaper Section 5)
```
IR(u) = I_min + (I_0 - I_min)/u_0 · u + (I_max - I_min - (I_0 - I_min)/u_0) · u^γ
```
- CLP Rate = u · IR(u) where u = utilization

---

## Detailed Sheet Formulas

### Input Variables

**BORROWER PARAMETERS:**
| Symbol | Description | Example |
|--------|-------------|---------|
| λ̃_e^C | Liquidation LTV of borrower's collateral (C) on Aave | 93% (e-mode) |
| λ̃_t^max | Twyne max Boosted Liquidation LTV | 98% |
| r_stake | Lending + Staking Rate (stETH yield) | 3.60% |
| r_borrow | Aave ETH Borrow Rate | 2.00% |

**CLP PARAMETERS:**
| Symbol | Description | Example |
|--------|-------------|---------|
| λ̃_e^CLP | Liquidation LTV of CLP collateral on Aave | 93% (e-mode) |
| β_safe | Safety buffer | 0.85-1.0 |
| IR(u) | Twyne credit delegation interest rate | varies |

**INTEREST RATE MODEL:**
| Symbol | Description | Example |
|--------|-------------|---------|
| I_min | Minimum interest rate | 0.00% |
| I_0 | Interest rate at kink | 0.80% |
| u_0 | Utilization at kink | 90.00% |
| I_max | Maximum interest rate | 20.00% |
| γ | Curvature parameter | 32.00 |

### Core Yield Formula

```
Y = (r_stake - r_borrow · λ_t - IR(u) · Ψ) / (1 - λ_t)
```

Where:
- λ_t = λ̃_t / HF (operating LTV derived from Health Factor)
- Ψ = λ̃_t / (β_safe · λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP

### Health Factor Decay (Days to Liquidation)

```
HF' = [(1 + r_stake - Ψ·IR(u)) / (1 + r_borrow)] · HF
    = [1 + r_net / (1 + r_borrow)] · HF
```

Where r_net = r_stake - r_borrow - Ψ·IR(u).

**Days to liquidation (when r_net < 0):**
```
T_liq = -365 · ln(HF_0) / ln(1 + r_net/(1 + r_borrow))
```

---

## Streamlit Dashboard (`app.py`)

### Features

**Tab 1: Detailed Sheet (Heatmaps)**
- Net Yield heatmap by HF × Boosted Liq LTV
- Days to Liquidation heatmap (purple = SAFE, when yield > 0)
- Current position marked with star

**Tab 2: Yield Calculator**
- Operating LTV, Boosted Liq LTV, CLP Cost Factor Ψ
- Interest Rate IR(u), Credit delegation cost, Net rate
- Yield vs Leverage curve

**Tab 3: Historical Analysis**
- Historical rates plot (stETH yield vs ETH borrow)
- Spread distribution histogram
- Leverage comparison (5x, 10x, 20x, 50x)

**Tab 4: Backtest**
- Run historical simulations at various leverage levels
- Capital and HF evolution over time
- Sharpe ratio, max drawdown, liquidation detection

### Sidebar Controls

**Position Settings:**
- Chain (Ethereum, Arbitrum, Optimism, Base, Polygon)
- Lending Market (Aave, Euler)
- CLP Asset (wstETH, WETH)
- Health Factor (1.01 - 1.20)
- Leverage (dynamic range)

**Rate Source:**
- Current (live from APIs) or Custom (manual input)

**External Market Parameters:**
- λ̃_e^C and λ̃_e^CLP sliders (defaults by chain/market)

**Twyne Parameters:**
- Max Boosted Liq LTV, Utilization, Safety Buffer
- Interest Rate Model parameters

---

## File Structure

```
boosted-looping-economics/
├── CLAUDE.md                              # This context file
├── README.md                              # Quick start guide
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore patterns
│
├── app.py                                 # Streamlit dashboard
├── twyne_looping/                         # Python module
│   ├── __init__.py                        # Module exports
│   ├── economics.py                       # Core financial formulas
│   ├── data.py                            # Data loading/preprocessing
│   ├── backtest.py                        # Backtesting engine
│   ├── visualizations.py                  # Matplotlib plots
│   └── live_rates.py                      # Aave/Lido API integration
│
├── data/
│   └── stETH-ETH_historical_APR_data.csv  # Historical rate data
│
├── notebooks/
│   └── analysis.ipynb                     # Exploratory analysis
│
└── docs/
    ├── README.md                          # Documentation index
    ├── Twyne_V1_Whitepaper.pdf            # Protocol documentation
    ├── Twyne_thread.pdf                   # Risk modeling analysis
    ├── Boosted_Looping_Economics.ods      # Economics spreadsheet
    └── technical-notes/
        ├── detailed_sheet_technical_doc.tex
        └── detailed_sheet_technical_doc.pdf
```

---

## Historical Data Statistics (2023-2025)

| Metric | Typical Range |
|--------|---------------|
| ETH Borrow Rate | 2-4% APR |
| stETH Supply Rate | 0-0.2% APR |
| Lido Staking Rate | 3-6% APR |
| Total stETH Yield | 3-6% APR |
| Net Spread | Usually positive (1-3%) |

---

## Risk Considerations

### For Borrowers (Loopers)
- Liquidation risk if spread turns negative for extended period
- Health factor degradation rate: r_net / (1 + r_borrow) per period
- Higher leverage = faster HF degradation when spread negative

### For Credit LPs
- Potential losses in fallback liquidation scenarios
- Protected by:
  - Large closing factors (50%+)
  - Low liquidation incentives
  - Safety buffer (β_safe)
- Expected annualized loss rate: ~0.16% at 94% max LLTV
- Break-even utilization: ~0.3%

---

*Last updated: 2026-01-06*
