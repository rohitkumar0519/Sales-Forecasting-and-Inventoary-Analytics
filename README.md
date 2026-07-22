# Project FORESIGHT
**AI-Powered Demand & Inventory Intelligence Platform**

An interactive dashboard that forecasts SKU-level demand, flags stockout and
overstock risk, and turns those signals into reorder / markdown
recommendations — built for supply chain and inventory planning teams.

## What's inside

| File | Purpose |
|---|---|
| `data_generator.py` | Generates realistic synthetic weekly sales history + inventory master data (25 SKUs × 3 sites, 2 years of history) |
| `forecasting.py` | Feature engineering + **XGBoost + LightGBM ensemble** forecasting, 12-week recursive horizon |
| `inventory.py` | Turns forecasts + stock levels into weeks-of-cover, risk classification, and reorder/markdown recommendations |
| `app.py` | Streamlit dashboard: Risk Radar grid, Forecast Explorer, Recommendations, Methodology |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. First load takes ~15–20 seconds
while the ensemble models train across the portfolio (cached after that).

## Swapping in real data

Replace the output of `load_all_data()` in `data_generator.py` with your own
extracts. You need three tables shaped like this:

- **sales_history**: `week_start, sku_id, store_id, units_sold, promotion_flag`
- **sku_master**: `sku_id, sku_name, category, unit_cost, price, lead_time_days`
- **inventory_master**: `sku_id, store_id, current_stock, avg_weekly_demand, safety_stock, reorder_point, lead_time_days, unit_cost, price, category, region, store_name, sku_name`

Everything downstream (`forecasting.py`, `inventory.py`, `app.py`) works
unchanged once those shapes match.

## Notes on the forecasting approach

Each SKU-site pair is modeled as its own weekly series with lag (1–12 wk),
rolling mean/std, and seasonal (sin/cos week-of-year) features. An XGBoost
regressor and a LightGBM regressor are trained independently and averaged;
their disagreement plus recent demand volatility form the confidence band.
Forecasts are generated recursively out to 12 weeks. For very short/sparse
histories, the model falls back to a naive seasonal-average forecast.
