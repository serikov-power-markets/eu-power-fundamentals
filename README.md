# Day-ahead power prices in Germany and France from fundamentals: a stack baseline, a delta model, nuclear outages and a spread signal check

Research study on public ENTSO-E data. What do the day-ahead forecasts of wind, solar and load, the gas and carbon price and the unavailability of French nuclear units say about the next day's baseload price in Germany-Luxembourg (DE) and France (FR) and about the DE-FR spread, and where does a simple fundamental stack stop and a data-driven delta model start? A walk-forward test over 2024 and 2025 (731 days, refitted monthly) with the information set of 10:00 on the day before delivery.

**Data (all public).** ENTSO-E Transparency Platform, January 2022 to December 2025: day-ahead prices (hourly means; the day-ahead market moved to 15-minute products in October 2025), day-ahead wind, solar and load forecasts, actual generation by type, and generation unit unavailability documents (type A80) for France, 22,267 documents, of which 5,627 unit periods concern nuclear units (57 units). TTF front-month daily close (Yahoo Finance series `TTF=F`, as in the sibling project `eu-gas-storage-valuation`) and the EEX auction price of EU allowances (daily, as in `ie-isem-market`; the EUA series ends in December 2025, which sets the end of the study period). Coal is not used (the monthly source ends earlier). The ENTSO-E physical flow series between DE and FR has no data on 638 of 1,761 days, so cross-border flows are not used as a feature.

## Results (`results/forecast_metrics.csv`, `results/spread_direction.csv`, `results/summary.json`)
Mean absolute error of the daily baseload price in EUR/MWh, test days 1 January 2024 to 31 December 2025:

| Model | DE | FR | DE-FR spread |
|---|---|---|---|
| Price of the previous delivery day | 20.6 | 15.5 | 16.0 |
| Stack: ridge on residual load, gas and carbon cost, French nuclear | 19.6 | 18.5 | 16.5 |
| Ridge on all fundamentals and price lags | 14.9 | 14.6 | 13.1 |
| Delta model: stack plus gradient boosting of its residual | 10.0 | 13.5 | 13.1 |
| Gradient boosting on all fundamentals and price lags | 8.9 | 12.4 | 13.7 |
| Gradient boosting on fundamentals only (no price lags) | 9.2 | 15.8 | 15.9 |
| Delta model plus French nuclear outages | 10.5 | 13.3 | 12.7 |
| Gradient boosting plus outages | 8.9 | 12.0 | 13.5 |

- **The linear stack is a weak baseline.** With five inputs it misses the level (bias -16 EUR/MWh for Germany) and is no better than yesterday's price. Adding the nonlinear delta term cuts the German error by half (19.6 to 10.0); for Germany the fundamentals alone (forecasts, load, fuel and carbon cost, no price lags) give 9.2, so almost everything comes from the fundamentals. For France the price lags matter (15.8 without, 12.4 with), consistent with a market driven by nuclear availability and hydro that the lags carry.
- **Nuclear outages help France and the spread, not Germany.** Known-at-the-gate unavailable nuclear capacity (planned and forced; mean 21.1 GW over the four years, 27.7 GW in 2022 against 17.3 in 2024 and 16.8 in 2025, 40.1 GW on 29 August 2022; the pattern fits the low French nuclear availability widely reported for 2022, which I did not check against RTE figures) lowers the French error of the boosting model from 12.35 to 11.96 EUR/MWh and the spread error of the delta model from 13.13 to 12.66; for Germany it does nothing (8.93 and 8.93; the delta model is slightly worse, 10.0 to 10.5).
- **Direction of the spread change (`spread_direction.csv`).** Against the previous delivery day's spread (mean absolute change 16.0 EUR/MWh) the delta model with outages gets the direction right on 78.0% of days, the delta model without outages 75.4%, a rule that reverts to the 7-day mean 63.9% and a rule that repeats the previous change 41.0%. The outage features win 34 days that the model without them loses and lose 15 (exact McNemar test p 0.009; several model comparisons were run and no correction was applied). This is information about direction, not a profit: there is no tradable price at the previous day's spread, so the mean gain of 11.5 EUR/MWh per day against that reference (after 0.5 EUR/MWh assumed cost) must not be read as a return, and no risk figure is given.
- **Monitoring (`rolling_mae_spread.csv`).** The error of the best spread model is 13.0 in 2024 and 12.3 in 2025; the 30-day mean error exceeds 1.5 times the 2024 level on 8 days of 2025. The population stability index of 2025 against the years before 2024 is large for the gas cost (4.4), the carbon price (2.9), French nuclear output (3.3) and the previous price (2.0), small for the German and French residual load forecasts (0.17 and 0.32): the fuel regime moved far from the training years, which the delta model has to absorb. A rule that raises an alert when the rolling error or a stability index exceeds a threshold is in `src/fund/models.py` (`psi`) and `scripts/run_backtest.py`.

## Information set and assumptions
Features for delivery day D are those available at 10:00 on D-1: day-ahead wind, solar and load forecasts for D (assumed published before the auction), prices of delivery days up to D-1 (published on D-2), generation and load of D-2 and earlier, TTF close and EUA price of D-2. **Outages (`src/fund/outages.py`).** The API returns only the final revision of each document, created when the outage ends, so the creation time cannot be used as the publication time (a first version that did so saw only 1.0 units out on average instead of about 21 GW here). The feature is therefore outages ongoing at the gate (started and not ended; assumed to last through D, the end date is not used) plus planned outages starting before the end of D (assumed announced in advance). The realised start of a forced outage that was only reported later is the remaining leak, so the gain from outages is an upper bound.

## What this does not show
No trading, no forward or intraday prices, no positions, so nothing about a P&L; no desk process, no live monitoring (the checks run on the historical test), no hourly or intraday shape, no coal price, no Switzerland, Belgium or Netherlands, no flow-based market coupling details, no reservoir hydro levels. Day-ahead forecasts are assumed to have been available at the stated time. One test period of two years with a fuel price regime far from the training years. The code is a research implementation, not a production system.

## Run
```
pip install pandas numpy scikit-learn scipy pyarrow requests pytest
python scripts/download.py        # ENTSO-E cache (needs a token, not stored in the repository)
python scripts/build_outages.py   # French nuclear unavailability known at the gate
python scripts/run_backtest.py
pytest
```
