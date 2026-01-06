# Boosted Looping Economics

Interactive dashboard and Python module for analyzing stETH-ETH leveraged yield farming economics using the Twyne credit delegation protocol.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Features

### Interactive Dashboard (`app.py`)

- **Detailed Sheet**: Heatmaps showing net yield and days-to-liquidation across different HF × Liq LTV combinations
- **Yield Calculator**: Calculate yields at specific leverage levels with full breakdown
- **Historical Analysis**: Visualize 2+ years of stETH/ETH rate history and spreads
- **Backtest**: Run historical simulations with capital and health factor tracking

### Python Module (`twyne_looping/`)

```python
from twyne_looping import (
    # Core economics
    looped_yield_from_leverage,
    days_to_liquidation,
    psi,
    interest_rate,

    # Data utilities
    load_apr_data,
    compute_spreads,

    # Backtesting
    run_backtest,
    BacktestResult,

    # Live rates
    fetch_live_rates,
    LiveRates,
)

# Calculate yield at 20x leverage
yield_20x = looped_yield_from_leverage(
    r_stake=0.04,      # 4% stETH yield
    r_borrow=0.02,     # 2% ETH borrow
    leverage=20,
    ir_u=0.008,        # Interest rate at 95% util
    psi_val=0.05,      # CLP cost factor
)
```

## Directory Structure

```
boosted-looping-economics/
├── app.py                    # Streamlit dashboard
├── twyne_looping/            # Python module
│   ├── economics.py          # Core financial formulas
│   ├── data.py               # Data loading utilities
│   ├── backtest.py           # Backtesting engine
│   ├── visualizations.py     # Plotting functions
│   └── live_rates.py         # Aave/Lido API integration
├── data/                     # Historical APR data
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation & research
│   ├── Twyne_V1_Whitepaper.pdf
│   ├── Boosted_Looping_Economics.ods
│   └── technical-notes/
└── CLAUDE.md                 # AI context document
```

## Key Concepts

### Twyne Boosted Leverage

Traditional Aave allows ~5x leverage on stETH-ETH (80% LTV). Twyne credit delegation enables up to 50x leverage by using Credit Liquidity Providers (CLPs) to supplement borrower collateral.

### Core Formula

Net yield at leverage L:
```
Y = (r_stake - r_borrow · λ_t - IR(u) · Ψ) / (1 - λ_t)
```

Where:
- `λ_t = (L-1)/L` is the operating LTV
- `Ψ` is the CLP cost factor (depends on boosted liq LTV)
- `IR(u)` is Twyne's interest rate at utilization u

### Health Factor Evolution

When spread is negative, health factor decays over time:
```
Days to liquidation = -365 · ln(HF₀) / ln(1 + r_net/(1 + r_borrow))
```

## Documentation

- `docs/Twyne_V1_Whitepaper.pdf` - Full protocol specification
- `docs/technical-notes/` - LaTeX derivations of all formulas
- `docs/Boosted_Looping_Economics.ods` - Reference spreadsheet
- `CLAUDE.md` - Comprehensive context for AI assistants

## Data

Historical APR data covers Jan 2023 - Oct 2025 with daily observations:
- ETH borrow rate (Aave)
- stETH supply rate (Aave)
- stETH staking rate (Lido)

## Contributing

See `CLAUDE.md` for technical context and formula derivations.
