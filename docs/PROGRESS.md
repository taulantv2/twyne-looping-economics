# Twyne Looping Economics Dashboard - Progress Notes

*Last updated: 2026-01-07*

---

## Recent Changes (Jan 7, 2026 - Session 2)

### 13. Color Scheme Update for Dark Mode

**Problem:** Aave colors (grey/black) weren't readable in dark mode.

**Solution:** Changed Aave colors to yellow/orange:
- 2x: `#fff176` (light yellow)
- 5x: `#ffee58` (yellow)
- 10x: `#fb8c00` (orange)
- 14x: `#ef6c00` (deep orange)

### 14. Legends Moved to Right Side

**Change:** All chart legends now positioned outside the plot on the right (`x=1.02`) to avoid obscuring data.

### 15. Results Summary Redesign

**Problem:** Old summary showed Min HF and Liquidated columns which didn't work/weren't useful.

**Solution:** New columns with progress bars and tooltips:
| Column | Description |
|--------|-------------|
| Leverage | Combined with protocol label (e.g., "Aave 14x") |
| Ann. Return | Annualized return with progress bar |
| Max Drawdown | Largest peak-to-trough decline |
| Return/Risk | Ann. Return ÷ Max Drawdown (higher = better) |
| Volatility | Annualized daily std dev |
| Sharpe | Risk-adjusted return |
| Win Rate | % of 30-day windows with positive returns |

All columns have hover tooltips explaining the metric.

### 16. Timing Sensitivity Redesign

**Problem:** Entry Date Sensitivity heatmaps showing "longer hold = better" wasn't insightful.

**Solution:** Two side-by-side charts:
1. **Return Range by Holding Period** (Convergence Cone): Shows best/worst/median returns with shaded range that narrows over time
2. **Loss Probability by Holding Period**: Shows % of entry dates that lost money, decreasing with hold time

**Features:**
- Granular data points every 7 days (52 points) for smooth hover
- Clean axis labels showing only key milestones (7, 30, 90, 180, 365)
- X-axis auto-truncates after loss probability reaches 0
- Legend groups link shaded range with median line for filtering

### 17. Performance Optimization

**Problem:** Timing sensitivity was slow with 52 holding periods.

**Solution:** Instead of running new backtests for each (leverage, holding period, entry date), now uses existing backtest capital data with vectorized pandas operations. ~100x faster.

### 18. Default Values Updated

- **Leverage levels**: Now `2, 14, 20, 30` (was `5, 10, 14, 20, 30`)
- **Health Factor**: Now allows 4 decimal places (`step=0.0001`, `format="%.4f"`)

### 19. Consistent Chart Explanations

**Problem:** Charts lacked consistent explanations, making dashboard hard to understand for new users.

**Solution:** Added `st.markdown("**Title**")` + `st.caption("explanation")` pattern to ALL charts:

| Chart | Explanation Added |
|-------|-------------------|
| Net Yield Heatmap | Theoretical yield at different HF/LTV. Green=profitable, Red=losing |
| Days to Liquidation | Time until HF=1.0. Purple=SAFE positions |
| Yield Calculator | How yield changes across LTVs at each HF |
| Value of Twyne | Is extra leverage worth credit cost? |
| Results Summary | Key performance metrics with progress bars |
| Capital Evolution | Growth of $1 at each leverage level |
| Rolling Yield | Return annualized, shows yield volatility |
| Drawdown | % below all-time high, shows worst losses |
| Timing Sensitivity | Entry timing effect on returns |
| ↳ Return Range | Shaded area = best/worst, line = median |
| ↳ Loss Probability | % of entry dates that lost money |
| Sharpe Ratio | Return/volatility. Above 1.0 = good |
| Return vs Risk | Scatter: ideal = top-left |
| Historical Rates | stETH yield vs ETH borrow over time |
| Spread Distribution | Histogram of daily spreads |
| Spread Risk Analysis | Tail risk metrics for stress testing |
| 30-Day Rolling Analysis | How often you'd lose money over a month |

---

## Recent Changes (Jan 7, 2026 - Session 1)

### 8. Consistent Color Scheme Across Charts

**Problem:** Historical Yield by Leverage chart used different colors than backtest charts.

**Solution:** Moved color scheme definitions to global scope:
- `AAVE_COLORS`: Grays for Aave-achievable leverage (≤14x)
- `TWYNE_COLORS`: Blues/purples for Twyne-boosted leverage (>14x)
- `get_leverage_style(L)`: Returns appropriate color based on leverage level

### 9. Tab Order Swap

**Problem:** Changing sidebar parameters caused tab reset to first tab (Yield Analysis), which was less useful during iteration.

**Solution:** Swapped tab order:
- Tab 1: Backtest & History (most frequently used)
- Tab 2: Yield Analysis

### 10. Returns by Holding Period Analysis

**Problem:** Entry Date Sensitivity heatmaps were hard to interpret.

**Solution:** Replaced with two side-by-side charts:
1. **Cumulative Returns by Holding Period**: Shows total return for each leverage level vs holding duration (7-365 days)
2. **Marginal Returns (Slope)**: Shows daily return rate at each holding period - useful for identifying diminishing returns

Both charts include 1x "Hold" benchmark (dashed green line) for comparison.

### 11. Fixed Axis Cutoff Issues

**Problem:** Sharpe Ratio and Return vs Risk plots had labels cut off.

**Solution:**
- Added 15% y-axis headroom for Sharpe chart
- Changed Return vs Risk text position to 'middle left'
- Added x/y axis padding for Risk-Return chart

### 12. Twyne Cost Conditional Application

**Problem:** Backtest applied Twyne credit delegation costs (IR(u) × Ψ) to ALL leverage levels, even those achievable on Aave alone.

**Solution:** Modified `backtest.py` to conditionally apply costs:
```python
aave_max_leverage = 1.0 / (1.0 - params["liq_ltv_c"])  # ~20x at 95% LLTV

if leverage > aave_max_leverage:
    psi_val = psi(...)
    ir_u = interest_rate(...)
else:
    psi_val = 0.0  # No Twyne costs for Aave-achievable leverage
    ir_u = 0.0
```

**Impact:** Lower leverage levels (5x, 10x, 20x) now show accurate returns without incorrectly deducted Twyne fees.

---

## Overview

This document tracks progress on the Streamlit dashboard (`app.py`) for analyzing stETH-ETH looping economics with Twyne protocol.

---

## Recent Changes (Jan 2026)

### 1. Historical Data Update

**Problem:** Historical APR data only extended to October 2025.

**Solution:** Updated data fetching to use DeFi Llama APIs:
- WETH pool: `e880e828-ca59-4ec6-8d4f-27182a4dc23d`
- wstETH pool: `e6435aae-cbe9-4d26-ab2c-a4d533db9972`
- Lido pool: `747c1d2a-c668-4682-b9f9-296708a3dd90`

**Result:** CSV now includes data through January 2026.

### 2. UI Simplification

**Changes made:**
- Removed chain selector (hardcoded to Ethereum)
- Removed lending market selector (hardcoded to Aave)
- Merged CLP and Collateral liquidation thresholds into single "Underlying LLTV" (assumes same for both)
- Added cache TTL of 60 seconds: `@st.cache_data(ttl=60)`

### 3. Position Controls Moved to Yield Calculator

**Problem:** Position parameters (Leverage, HF) were in sidebar but only relevant for Yield Calculator tab.

**Solution:** Moved position controls into Tab 2 (Yield Calculator) only.

**Current sidebar structure:**
```
⚙️ Parameters
├── Rate Source (Current/Custom)
│   └── Live rates display or manual sliders
├── Raw Spread display
├── [Expander] Liquidation Thresholds
│   ├── Underlying LLTV (λ̃_e)
│   └── Twyne Max LLTV (λ̃_t^max)
└── [Expander] Advanced
    ├── CLP Utilization
    ├── Safety Buffer (β_safe)
    └── Interest Rate Model (I_min, I_0, u_0, I_max, γ)
```

### 4. Detailed Sheet (Tab 1) Cleanup

**Changes:**
- Removed position star marker from yield heatmap
- Removed position star marker from DTL heatmap
- DTL plot now shows only when `np.any(yield_grid < 0)` (any negative yields in grid)
- Fixed HF range to 1.0-1.2 (no longer depends on position)
- LTV range: 1% below Aave LLTV to Twyne max LLTV

### 5. Leverage/HF Constraint Fix

**Problem:** Users could set positions where λ̃_t exceeded λ̃_t^max (would be liquidated).

**Solution:** Dynamic HF slider max based on leverage:
```python
lambda_t = (leverage - 1) / leverage
HF_max = liq_ltv_max / lambda_t
hf_slider_max = max(1.01, floor(HF_max * 100) / 100)
```

This ensures `λ̃_t = HF × (L-1)/L ≤ λ̃_t^max` always holds.

### 6. Twyne Interest Rate Model Parameters

**Corrected values:**
| Parameter | Value |
|-----------|-------|
| I_min | 0.0% |
| I_0 | 0.8% |
| u_0 | 90% |
| I_max | 0.5% |
| γ | 32 |
| β_safe | 1.0 |
| λ̃_t^max | 98% |

### 7. Heatmap Improvements

- Added Aave threshold vertical line (blue dashed) at underlying LLTV
- Using RdYlGn colorscale with `zmid=0` (negative=red, positive=green)
- Numeric x-axis values for proper line positioning
- Yield values available in expandable table below heatmap

---

## Current Tab Structure

### Tab 1: Backtest & History
- **Backtest Settings**: Leverage levels (default: 2, 14, 20, 30), initial HF (4 decimals), date range
- **Results Summary**: Table with progress bars (Ann. Return, Max Drawdown, Return/Risk, Volatility, Sharpe, Win Rate)
- **Capital Evolution**: Line chart showing portfolio growth per leverage
- **Rolling Annualized Yield**: Selectable window (Daily, 7 Days, 30 Days)
- **Drawdown**: Peak-to-trough decline visualization
- **Timing Sensitivity**: Side-by-side Convergence Cone + Loss Probability charts
- **Risk-Adjusted Performance**: Sharpe Ratio and Return vs Risk scatter
- **Historical Rates**: stETH yield vs ETH borrow over time
- **Historical Yield by Leverage**: Multi-leverage yield comparison

### Tab 2: Yield Analysis
- **Detailed Sheet**: Net Yield heatmap (HF × Boosted Liq LTV)
- Days to Liquidation heatmap (shown only if negative yields exist)
- Aave threshold line at underlying LLTV
- **Yield Calculator**: Position controls (Leverage, HF)
- Derived metrics (Operating LTV, Boosted Liq LTV, Ψ, IR(u), etc.)
- Yield vs Leverage curve
- Yield breakdown table

---

## Key Formulas Reference

**Leverage ↔ LTV:**
```
L = 1/(1 - λ_t)
λ_t = (L-1)/L
```

**Health Factor:**
```
HF = λ̃_t / λ_t
λ̃_t = HF × λ_t
```

**CLP Cost Factor (Ψ):**
```
Ψ = λ̃_t / (β_safe × λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP
```

**Net Yield:**
```
Y = (r_stake - r_borrow × λ_t - IR(u) × Ψ) / (1 - λ_t)
```

**Interest Rate Model:**
```
IR(u) = I_min + (I_0 - I_min)/u_0 × u + (I_max - I_min - (I_0 - I_min)/u_0) × u^γ
```

---

## Known Issues / Future Work

1. **Deprecation warnings:** Fixed `use_container_width` → `width='stretch'`

2. **Data freshness:** Consider automating historical data updates

3. **Multi-chain support:** Currently hardcoded to Ethereum/Aave. Could re-enable chain selection if needed.

4. **Live rates caching:** Currently fetches on each page load. Consider longer cache for API rate limits.

---

## File Structure

```
app.py                              # Main Streamlit dashboard
twyne_looping/
├── __init__.py
├── economics.py                    # Core financial formulas
├── data.py                         # Data loading
├── backtest.py                     # Backtesting engine
├── visualizations.py               # Matplotlib plots (unused in Streamlit)
└── live_rates.py                   # Aave/Lido API integration
data/
└── stETH-ETH_historical_APR_data.csv
docs/
├── PROGRESS.md                     # This file
├── Twyne_V1_Whitepaper.pdf
├── Twyne_thread.pdf
├── Boosted_Looping_Economics.ods
└── technical-notes/
```

---

## Running the App

```bash
# Using uv (recommended)
uv run streamlit run app.py

# Or with pip
pip install -r requirements.txt
streamlit run app.py
```

App runs at: http://localhost:8501 (or next available port)
