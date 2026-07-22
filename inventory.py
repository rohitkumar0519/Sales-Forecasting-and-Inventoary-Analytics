"""
Project FORESIGHT - Inventory Intelligence Layer
Turns forecasts + current stock into stockout/overstock risk scores
and actionable reorder / markdown recommendations.
"""

import numpy as np
import pandas as pd


def weeks_of_cover(current_stock: float, avg_forecast_weekly: float) -> float:
    if avg_forecast_weekly <= 0:
        return np.inf
    return current_stock / avg_forecast_weekly


def classify_risk(weeks_cover: float, lead_time_weeks: float) -> str:
    """
    Stockout risk: cover is less than lead time -> will likely run out before
    replenishment arrives.
    Overstock risk: cover is more than 3x the "healthy" band.
    """
    if weeks_cover < lead_time_weeks * 1.1:
        return "Stockout Risk"
    elif weeks_cover > lead_time_weeks * 4:
        return "Overstock Risk"
    else:
        return "Healthy"


def build_inventory_intelligence(inventory_master: pd.DataFrame, forecasts: dict) -> pd.DataFrame:
    """
    forecasts: dict keyed by (sku_id, store_id) -> forecast DataFrame from forecasting.py
    Returns inventory_master enriched with forecasted demand, risk classification,
    and recommended order quantity.
    """
    rows = []
    for _, row in inventory_master.iterrows():
        key = (row["sku_id"], row["store_id"])
        fc = forecasts.get(key)
        if fc is None or fc.empty:
            avg_forecast = row["avg_weekly_demand"]
            next4_forecast = avg_forecast * 4
        else:
            avg_forecast = fc["forecast"].mean()
            next4_forecast = fc["forecast"].head(4).sum()

        lead_time_weeks = row["lead_time_days"] / 7
        cover = weeks_of_cover(row["current_stock"], avg_forecast)
        risk = classify_risk(cover, lead_time_weeks)

        # Recommended order-up-to level: cover lead time + safety stock + 4 weeks buffer
        target_level = avg_forecast * (lead_time_weeks + 2) + row["safety_stock"]
        recommended_order_qty = max(round(target_level - row["current_stock"]), 0) if risk == "Stockout Risk" else 0

        excess_units = max(round(row["current_stock"] - target_level), 0) if risk == "Overstock Risk" else 0
        potential_markdown_value = round(excess_units * row["unit_cost"], 2)

        stockout_cost_estimate = round(
            max(next4_forecast - row["current_stock"], 0) * (row["price"] - row["unit_cost"]), 2
        ) if risk == "Stockout Risk" else 0.0

        rows.append({
            **row.to_dict(),
            "forecast_avg_weekly": round(avg_forecast, 1),
            "forecast_next_4wk": round(next4_forecast, 1),
            "weeks_of_cover": round(cover, 1) if np.isfinite(cover) else 999,
            "risk_status": risk,
            "recommended_order_qty": int(recommended_order_qty),
            "excess_units": int(excess_units),
            "potential_markdown_value": potential_markdown_value,
            "stockout_lost_sales_estimate": stockout_cost_estimate,
        })

    return pd.DataFrame(rows)


def portfolio_kpis(intel_df: pd.DataFrame) -> dict:
    total_skus_locations = len(intel_df)
    stockout_count = (intel_df["risk_status"] == "Stockout Risk").sum()
    overstock_count = (intel_df["risk_status"] == "Overstock Risk").sum()
    healthy_count = (intel_df["risk_status"] == "Healthy").sum()

    total_inventory_value = (intel_df["current_stock"] * intel_df["unit_cost"]).sum()
    total_markdown_exposure = intel_df["potential_markdown_value"].sum()
    total_lost_sales_risk = intel_df["stockout_lost_sales_estimate"].sum()

    return {
        "total_sku_locations": total_skus_locations,
        "stockout_count": int(stockout_count),
        "overstock_count": int(overstock_count),
        "healthy_count": int(healthy_count),
        "stockout_pct": round(100 * stockout_count / total_skus_locations, 1) if total_skus_locations else 0,
        "overstock_pct": round(100 * overstock_count / total_skus_locations, 1) if total_skus_locations else 0,
        "total_inventory_value": round(total_inventory_value, 2),
        "total_markdown_exposure": round(total_markdown_exposure, 2),
        "total_lost_sales_risk": round(total_lost_sales_risk, 2),
    }
