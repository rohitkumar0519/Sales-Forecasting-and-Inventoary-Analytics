"""
Project FORESIGHT - Synthetic Data Generator
Generates realistic SKU-level sales history + inventory master data
so the platform can be demoed end-to-end without a live data source.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CATEGORIES = {
    "Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank", "USB-C Cable"],
    "Home & Kitchen": ["Air Fryer", "Blender", "Coffee Maker", "Vacuum Cleaner", "Toaster"],
    "Apparel": ["Running Shoes", "Denim Jacket", "Cotton T-Shirt", "Winter Hoodie", "Rain Jacket"],
    "Personal Care": ["Electric Toothbrush", "Hair Dryer", "Face Serum", "Shaving Kit", "Body Lotion"],
    "Grocery": ["Organic Coffee", "Protein Bar Pack", "Olive Oil 1L", "Green Tea Box", "Almond Butter"],
}

STORES = [
    {"store_id": "ST01", "region": "North", "store_name": "Central Warehouse"},
    {"store_id": "ST02", "region": "South", "store_name": "Southgate DC"},
    {"store_id": "ST03", "region": "West", "store_name": "Westline Hub"},
]

WEEKS_OF_HISTORY = 104  # 2 years of weekly history


def _build_sku_master():
    rows = []
    sku_num = 1000
    for category, products in CATEGORIES.items():
        for product in products:
            sku_num += 1
            base_demand = RNG.integers(40, 400)
            seasonality_strength = RNG.uniform(0.05, 0.45)
            trend = RNG.uniform(-0.15, 0.35)
            unit_cost = round(RNG.uniform(4, 120), 2)
            price = round(unit_cost * RNG.uniform(1.4, 2.6), 2)
            lead_time_days = int(RNG.choice([5, 7, 10, 14, 21]))
            rows.append({
                "sku_id": f"SKU{sku_num}",
                "sku_name": product,
                "category": category,
                "base_demand": base_demand,
                "seasonality_strength": seasonality_strength,
                "trend": trend,
                "unit_cost": unit_cost,
                "price": price,
                "lead_time_days": lead_time_days,
            })
    return pd.DataFrame(rows)


def _simulate_weekly_sales(sku_row, store_id, n_weeks=WEEKS_OF_HISTORY):
    t = np.arange(n_weeks)
    base = sku_row["base_demand"]
    trend_component = sku_row["trend"] * t / 10
    seasonal_component = base * sku_row["seasonality_strength"] * np.sin(2 * np.pi * t / 52)
    holiday_bump = np.where((t % 52 >= 47) | (t % 52 <= 1), base * 0.6, 0)  # year-end bump
    noise = RNG.normal(0, base * 0.12, size=n_weeks)
    promo_flags = RNG.choice([0, 1], size=n_weeks, p=[0.85, 0.15])
    promo_bump = promo_flags * base * RNG.uniform(0.3, 0.8)

    demand = base + trend_component + seasonal_component + holiday_bump + promo_bump + noise
    demand = np.clip(demand, 0, None).round().astype(int)

    store_factor = {"ST01": 1.0, "ST02": 0.75, "ST03": 0.55}[store_id]
    demand = np.round(demand * store_factor).astype(int)

    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(weeks=n_weeks)
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

    return pd.DataFrame({
        "week_start": dates,
        "sku_id": sku_row["sku_id"],
        "store_id": store_id,
        "units_sold": demand,
        "promotion_flag": promo_flags,
    })


def generate_sales_history():
    sku_master = _build_sku_master()
    frames = []
    for _, sku_row in sku_master.iterrows():
        for store in STORES:
            frames.append(_simulate_weekly_sales(sku_row, store["store_id"]))
    sales = pd.concat(frames, ignore_index=True)
    return sales, sku_master


def generate_inventory_master(sku_master, sales_history):
    rows = []
    for _, sku in sku_master.iterrows():
        for store in STORES:
            recent = sales_history[
                (sales_history["sku_id"] == sku["sku_id"]) & (sales_history["store_id"] == store["store_id"])
            ].tail(8)
            avg_weekly_demand = max(recent["units_sold"].mean(), 1)
            lead_time_weeks = sku["lead_time_days"] / 7
            safety_stock = round(avg_weekly_demand * RNG.uniform(0.5, 1.2), 0)
            reorder_point = round(avg_weekly_demand * lead_time_weeks + safety_stock, 0)

            # Deliberately scatter stock levels around reorder point so risk categories emerge
            stock_scenario = RNG.choice(["low", "normal", "high"], p=[0.22, 0.56, 0.22])
            if stock_scenario == "low":
                current_stock = round(reorder_point * RNG.uniform(0.15, 0.85))
            elif stock_scenario == "high":
                current_stock = round(reorder_point * RNG.uniform(2.2, 4.0))
            else:
                current_stock = round(reorder_point * RNG.uniform(0.9, 1.8))

            rows.append({
                "sku_id": sku["sku_id"],
                "sku_name": sku["sku_name"],
                "category": sku["category"],
                "store_id": store["store_id"],
                "region": store["region"],
                "store_name": store["store_name"],
                "current_stock": int(max(current_stock, 0)),
                "avg_weekly_demand": round(avg_weekly_demand, 1),
                "safety_stock": int(safety_stock),
                "reorder_point": int(reorder_point),
                "lead_time_days": sku["lead_time_days"],
                "unit_cost": sku["unit_cost"],
                "price": sku["price"],
            })
    return pd.DataFrame(rows)


def load_all_data():
    """Single entry point used by the app. Returns (sales_history, sku_master, inventory_master)."""
    sales_history, sku_master = generate_sales_history()
    inventory_master = generate_inventory_master(sku_master, sales_history)
    return sales_history, sku_master, inventory_master


if __name__ == "__main__":
    sales, sku, inv = load_all_data()
    print("Sales history:", sales.shape)
    print("SKU master:", sku.shape)
    print("Inventory master:", inv.shape)
    print(sales.head())
    print(inv.head())
