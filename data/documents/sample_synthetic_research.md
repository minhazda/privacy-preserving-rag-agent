# Synthetic Research Summary (Sample Corpus)

> This is a **synthetic, illustrative** summary used so the RAG pipeline works
> out of the box and in CI. Replace/augment it by dropping the real
> `dissertation.pdf` and `preprint.pdf` into this folder (they are gitignored).

## Topic
Real-Time Retail Demand Forecasting Using Synthetic Data. The work studies
hourly, per-SKU demand forecasting trained entirely on privacy-preserving
synthetic data, removing any dependency on real customer records.

## Method
A seeded synthetic generator produces hourly sales for 100 SKUs across five
categories, with calendar, weather, promotion, and foot-traffic signals. A
shared feature pipeline adds per-SKU lag, rolling-mean, and one-hot features
with strict no-look-ahead leakage guards. A LightGBM regressor is trained on a
time-ordered split.

## Key Results
On a held-out time split the LightGBM model reduces MAE by roughly 40% and RMSE
by roughly 47% versus a seasonal-naive (previous-day, same-hour) baseline.
Absolute MAPE is high because the target is a low, sparse hourly count, so the
baseline-relative reduction is the meaningful, scale-robust metric.

## Privacy Position
Because all data is synthetic, no real personal or customer information is ever
processed or exposed. The serving layer additionally enforces a synthetic-only
allow-list and redacts any PII-shaped strings as defence in depth.
