# Monte-Carlo-Value-at-Risk-Engine
Monte Carlo Value-at-Risk engine for FTSE 100 data (2018–2025), combining PostfreSQL for data preperation , GPU-accelerated PyTorch simulation, and EVT tail modelling and  validated on the 2020 COVID crash

# Monte Carlo VaR Engine (Independent Research)

Period: Aug 2025 – Sep 2025  
Author: Trevor Higgins   

## Overview
The project implements a Monte Carlo Value-at-Risk (VaR) engine for the FTSE 100 index data from 2018–2025.  
It utilises SQL-based data preparation, PyTorch Monte Carlo simulation (100K+ paths, GPU-enabled),  
and Extreme Value Theory (EVT) tail modelling to produce the robust estimates of downside risk  

The engine was validated against the 2020 COVID crash, with it demonstrating an accuracy at the 99%–99.5% levels.  

## Workflow
1. **Data Preparation**  
   - Source: FTSE 100 historical data from [Investing.com](https://uk.investing.com/indices/uk-100-historical-data?cid=27517).  
   - Cleaned using SQL CTEs and window functions (`LAG`, `AVG`) to compute daily returns and moving averages.  

2. **Baseline Risk Estimation**  
   - Historical Simulation VaR (HS) for 95% and 99%.  
   - Limitations: underestimates tails, insensitive to volatility shifts.  

3. **Monte Carlo Simulation**  
   - PyTorch-based Normal MC (100K+ paths).  
   - GPU acceleration for scalable simulations.  

4. **Extreme Value Theory (EVT) Integration**  
   - Peaks-over-threshold method (5% left tail).  
   - Generalized Pareto Distribution (GPD) fit.  
   - EVT samples spliced into MC tail → fatter, more realistic losses.  

5. **Backtesting (COVID Crash: Feb–Jun 2020)**  
   - Compared realised FTSE 100 losses to simulated VaR thresholds.  
   - The Results were as followed:  
     - 95% EVT VaR → conservative (too many breaches).  
     - 99% EVT VaR → ~1.9% error (very close).  
     - 99.5% EVT VaR → perfect alignment (1 breach vs 1 expected).  

## Key Results

| Method                 | 95% VaR | 99% VaR | 99.5% VaR | COVID Backtest Accuracy  |
|------------------------|---------|---------|-----------|--------------------------|
| Historical Simulation  | Underfit| Conservative | N/A  | Misaligned tails         |
| Normal Monte Carlo     | ~-1.6%  | ~-2.3%  | ~-2.5%    | Underestimates tails     |
| EVT-Adjusted MC (final)| ~-1.7%  | ~-3.8%  | ~-9.4%    | ≤1.9% error at 99%, perfect at 99.5% |

*Portfolio size was £10M notional; horizon: 1-day.*

## Installation
```bash
git clone <your-repo>
cd mc-var-engine
pip install -r requirements.txt
