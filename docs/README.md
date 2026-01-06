# Documentation

Reference materials and technical documentation for the Boosted Looping Economics project.

## Contents

### Protocol Documentation

- **`Twyne_V1_Whitepaper.pdf`** - Comprehensive 32-page whitepaper covering:
  - Credit delegation mechanics
  - Dual LTV framework
  - Liquidation processes
  - Interest rate model
  - Protocol invariants

### Research & Analysis

- **`Twyne_thread.pdf`** - Risk modeling analysis based on 1,115 Euler V2 liquidation events (Sept 2024 - May 2025)
  - Key finding: ~0.16% annualized CLP loss rate at 94% max LLTV
  - Break-even utilization: ~0.3%

### Reference Spreadsheet

- **`Boosted_Looping_Economics.ods`** - LibreOffice Calc spreadsheet containing:
  - "Detailed Sheet" with all economics formulas
  - Historical APR data analysis
  - Parameter sensitivity studies

### Technical Notes

- **`technical-notes/detailed_sheet_technical_doc.tex`** - LaTeX source with complete mathematical derivations
- **`technical-notes/detailed_sheet_technical_doc.pdf`** - Compiled PDF documentation

## Key Formulas

### Net Yield
```
Y = (r_stake - r_borrow · λ_t - IR(u) · Ψ) / (1 - λ_t)
```

### CLP Cost Factor (Ψ)
```
Ψ = λ̃_t / (β_safe · λ̃_e^CLP) - λ̃_e^C / λ̃_e^CLP
```

### Days to Liquidation
```
T_liq = -365 · ln(HF_0) / ln(1 + r_net/(1 + r_borrow))
```

See `technical-notes/` for complete derivations.
