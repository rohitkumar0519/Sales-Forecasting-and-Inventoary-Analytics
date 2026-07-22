"""
Project FORESIGHT - Forecasting Engine
Ensemble of XGBoost + LightGBM regressors on engineered lag/rolling/seasonal
features, trained per SKU-store series, with recursive multi-step forecasting.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

FORECAST_HORIZON_WEEKS = 12
LAGS = [1, 2, 3, 4, 8, 12]
ROLLING_WINDOWS = [4, 8, 12]


def _build_features(series: pd.DataFrame) -> pd.DataFrame:
    """series must have columns: week_start, units_sold, promotion_flag (sorted by week_start)."""
    df = series.copy().reset_index(drop=True)
    for lag in LAGS:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = df["units_sold"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["units_sold"].shift(1).rolling(window).std()
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + ["sin_week", "cos_week", "month", "promotion_flag"]
)


def _train_ensemble(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLS]
    y = train_df["units_sold"]

    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    lgbm = LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    xgb.fit(X, y)
    lgbm.fit(X, y)
    return xgb, lgbm


def forecast_sku_store(series: pd.DataFrame, horizon: int = FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
    """
    Trains an XGBoost + LightGBM ensemble on one SKU-store weekly series and
    recursively forecasts `horizon` weeks ahead.
    Returns a DataFrame with week_start, forecast, forecast_lower, forecast_upper.
    """
    feat_df = _build_features(series)
    train_df = feat_df.dropna(subset=FEATURE_COLS).copy()

    if len(train_df) < 20:
        # Not enough history for ML -> fall back to naive seasonal-mean forecast
        avg = series["units_sold"].tail(8).mean()
        future_dates = pd.date_range(
            series["week_start"].max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON"
        )
        return pd.DataFrame({
            "week_start": future_dates,
            "forecast": [max(avg, 0)] * horizon,
            "forecast_lower": [max(avg * 0.7, 0)] * horizon,
            "forecast_upper": [avg * 1.3] * horizon,
        })

    xgb, lgbm = _train_ensemble(train_df)

    history = series[["week_start", "units_sold", "promotion_flag"]].copy()
    future_rows = []
    last_date = history["week_start"].max()

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(weeks=1)
        promo_flag = 0  # assume no promo planned unless specified
        working = pd.concat([
            history,
            pd.DataFrame([{"week_start": next_date, "units_sold": np.nan, "promotion_flag": promo_flag}])
        ], ignore_index=True)

        feat = _build_features(working).iloc[[-1]]
        X_next = feat[FEATURE_COLS]

        pred_xgb = xgb.predict(X_next)[0]
        pred_lgbm = lgbm.predict(X_next)[0]
        pred = max((pred_xgb + pred_lgbm) / 2, 0)

        # simple uncertainty band from ensemble spread + recent volatility
        recent_std = history["units_sold"].tail(8).std()
        recent_std = 0 if np.isnan(recent_std) else recent_std
        spread = abs(pred_xgb - pred_lgbm)
        band = max(spread, recent_std * 0.5, pred * 0.1)

        future_rows.append({
            "week_start": next_date,
            "forecast": round(pred, 1),
            "forecast_lower": round(max(pred - band, 0), 1),
            "forecast_upper": round(pred + band, 1),
        })

        history = pd.concat([
            history,
            pd.DataFrame([{"week_start": next_date, "units_sold": pred, "promotion_flag": promo_flag}])
        ], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(future_rows)


def forecast_multiple(sales_history: pd.DataFrame, sku_id: str, store_id: str, horizon: int = FORECAST_HORIZON_WEEKS):
    series = sales_history[
        (sales_history["sku_id"] == sku_id) & (sales_history["store_id"] == store_id)
    ].sort_values("week_start")
    return forecast_sku_store(series, horizon=horizon)


if __name__ == "__main__":
    from data_generator import load_all_data
    sales, sku_master, inv = load_all_data()
    sample_sku = sku_master.iloc[0]["sku_id"]
    fc = forecast_multiple(sales, sample_sku, "ST01")
    print(fc)
