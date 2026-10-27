"""
Machine Learning prediction engine for Crypto Forecasting.
Uses Gradient Boosting with lag features and rolling statistics.
Trains separate models for price, volume, and market_cap.
"""
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import database as db
from config import (
    CRYPTO_ASSETS, PREDICTION_HORIZONS, MODEL_DIR, MIN_TRAINING_ROWS
)

os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["price_usd", "volume_24h", "market_cap"]

# Lag periods (days) to create features from
LAG_DAYS = [1, 2, 3, 5, 7, 14, 21, 30]
ROLLING_WINDOWS = [3, 7, 14, 30]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame, target_col: str, horizon: int) -> tuple:
    """
    Build feature matrix X and target vector y from price history.

    Returns (X, y, feature_names)  or (None, None, None) if not enough data.
    """
    df = df.sort_values("timestamp").copy()
    df = df.dropna(subset=[target_col])

    if len(df) < MIN_TRAINING_ROWS + horizon:
        return None, None, None

    # --- base features ---
    for col in ["price_usd", "volume_24h", "market_cap"]:
        if col not in df.columns:
            df[col] = np.nan

    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["day_of_month"] = pd.to_datetime(df["timestamp"]).dt.day
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month

    feat_cols = ["day_of_week", "day_of_month", "month"]

    for col in ["price_usd", "volume_24h", "market_cap"]:
        series = df[col].ffill()

        # Lag features
        for lag in LAG_DAYS:
            name = f"{col}_lag{lag}"
            df[name] = series.shift(lag)
            feat_cols.append(name)

        # Rolling statistics
        for win in ROLLING_WINDOWS:
            df[f"{col}_roll_mean{win}"] = series.shift(1).rolling(win).mean()
            df[f"{col}_roll_std{win}"]  = series.shift(1).rolling(win).std()
            feat_cols.extend([f"{col}_roll_mean{win}", f"{col}_roll_std{win}"])

        # Rate of change
        df[f"{col}_roc7"]  = series.pct_change(7)
        df[f"{col}_roc14"] = series.pct_change(14)
        feat_cols.extend([f"{col}_roc7", f"{col}_roc14"])

    # Target: value `horizon` days into the future
    df["target"] = df[target_col].shift(-horizon)

    df = df.dropna(subset=feat_cols + ["target"])

    X = df[feat_cols].values
    y = df["target"].values
    return X, y, feat_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(symbol: str, target: str, horizon: int) -> dict:
    """
    Train a Gradient Boosting model for the given symbol/target/horizon.
    Returns metrics dict.
    """
    print(f"[ML] Training {symbol} | {target} | {horizon}d horizon...")

    df = db.get_price_history(symbol, days=730)
    if df.empty:
        print(f"[ML] No data for {symbol}")
        return {}

    X, y, feat_cols = _build_features(df, target, horizon)
    if X is None:
        print(f"[ML] Not enough data for {symbol}/{target}/{horizon}d "
              f"(need {MIN_TRAINING_ROWS + horizon} rows, have {len(df)})")
        return {}

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_mae, cv_rmse = [], []

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_split=10,
        random_state=42,
    )

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        gb.fit(X_tr, y_tr)
        preds = gb.predict(X_val)
        cv_mae.append(mean_absolute_error(y_val, preds))
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))

    # Final fit on all data
    gb.fit(X, y)
    final_preds = gb.predict(X)
    r2 = r2_score(y, final_preds)
    mae = float(np.mean(cv_mae))
    rmse = float(np.mean(cv_rmse))

    # Persist model
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    meta = {
        "model": gb,
        "feat_cols": feat_cols,
        "symbol": symbol,
        "target": target,
        "horizon": horizon,
        "trained_at": datetime.utcnow().isoformat(),
    }
    joblib.dump(meta, model_path)

    # Save metrics to DB
    with db.get_connection() as conn:
        cursor = conn.cursor()
        # deactivate old
        cursor.execute("""
            UPDATE ml_models SET is_active = 0
            WHERE symbol = ? AND target = ? AND horizon_days = ?
        """, symbol, target, horizon)
        cursor.execute("""
            INSERT INTO ml_models
                (symbol, target, horizon_days, model_file, mae, rmse,
                 r2_score, training_rows, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, symbol, target, horizon, model_path, mae, rmse, r2, len(y))
        model_id = cursor.execute("SELECT @@IDENTITY").fetchone()[0]

    metrics = {
        "symbol": symbol, "target": target, "horizon": horizon,
        "mae": mae, "rmse": rmse, "r2": r2,
        "rows": len(y), "model_path": model_path,
    }
    print(f"[ML] {symbol}/{target}/{horizon}d | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")
    return metrics


def train_all():
    """Train all models for all symbols, targets, and horizons."""
    results = []
    for symbol in CRYPTO_ASSETS:
        for target in TARGETS:
            for horizon in PREDICTION_HORIZONS:
                try:
                    m = train_model(symbol, target, horizon)
                    if m:
                        results.append(m)
                except Exception as e:
                    print(f"[ML] ERROR training {symbol}/{target}/{horizon}d: {e}")
    print(f"[ML] Training complete. {len(results)} models trained.")
    return results


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _load_model(symbol: str, target: str, horizon: int):
    """Load model from disk."""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict(symbol: str, target: str, horizon: int) -> dict:
    """
    Generate a prediction for the given symbol/target/horizon.
    Returns dict with predicted_value, lower, upper, predicted_for.
    """
    meta = _load_model(symbol, target, horizon)
    if meta is None:
        return None

    df = db.get_price_history(symbol, days=120)
    if df.empty or len(df) < 35:
        return None

    # Build the same features on the latest available row
    df = df.sort_values("timestamp").copy()

    # We need to replicate feature engineering on the tail of the DF
    for col in ["price_usd", "volume_24h", "market_cap"]:
        if col not in df.columns:
            df[col] = np.nan

    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["day_of_month"] = pd.to_datetime(df["timestamp"]).dt.day
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month

    for col in ["price_usd", "volume_24h", "market_cap"]:
        series = df[col].ffill()
        for lag in LAG_DAYS:
            df[f"{col}_lag{lag}"] = series.shift(lag)
        for win in ROLLING_WINDOWS:
            df[f"{col}_roll_mean{win}"] = series.shift(1).rolling(win).mean()
            df[f"{col}_roll_std{win}"]  = series.shift(1).rolling(win).std()
        df[f"{col}_roc7"]  = series.pct_change(7)
        df[f"{col}_roc14"] = series.pct_change(14)

    feat_cols = meta["feat_cols"]
    model = meta["model"]

    row = df[feat_cols].iloc[-1]
    if row.isnull().all():
        return None

    row_filled = row.fillna(0).values.reshape(1, -1)
    pred_value = float(model.predict(row_filled)[0])

    # Confidence interval: use MAE from training as proxy
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TOP 1 mae FROM ml_models
                WHERE symbol=? AND target=? AND horizon_days=? AND is_active=1
                ORDER BY trained_at DESC
            """, symbol, target, horizon)
            row_db = cursor.fetchone()
            margin = row_db[0] if row_db else pred_value * 0.05
    except Exception:
        margin = pred_value * 0.05

    predicted_for = (datetime.utcnow() + timedelta(days=horizon)).replace(
        hour=0, minute=0, second=0, microsecond=0)

    result = {
        "symbol": symbol,
        "target": target,
        "horizon": horizon,
        "predicted_for": predicted_for,
        "predicted_value": pred_value,
        "lower_bound": pred_value - margin,
        "upper_bound": pred_value + margin,
    }

    # Persist to DB
    try:
        db.save_prediction(
            symbol=symbol,
            target=target,
            horizon=horizon,
            predicted_for=predicted_for,
            value=pred_value,
            lower=result["lower_bound"],
            upper=result["upper_bound"],
        )
    except Exception as e:
        print(f"[ML] Could not save prediction: {e}")

    return result


def predict_all() -> list:
    """Generate and store predictions for all symbols/targets/horizons."""
    results = []
    for symbol in CRYPTO_ASSETS:
        for target in TARGETS:
            for horizon in PREDICTION_HORIZONS:
                try:
                    r = predict(symbol, target, horizon)
                    if r:
                        results.append(r)
                except Exception as e:
                    print(f"[ML] Predict error {symbol}/{target}/{horizon}d: {e}")
    print(f"[ML] Generated {len(results)} predictions.")
    return results


def get_prediction_summary() -> dict:
    """
    Return a nested dict of predictions:
    { symbol -> { horizon -> { target -> {value, lower, upper, predicted_for} } } }
    """
    df = db.get_latest_predictions()
    summary = {}
    for _, row in df.iterrows():
        sym  = row["symbol"]
        hor  = str(int(row["horizon_days"]))  # string key so JSON roundtrip is consistent
        tgt  = row["target"]
        summary.setdefault(sym, {}).setdefault(hor, {})[tgt] = {
            "value": row["predicted_value"],
            "lower": row.get("lower_bound"),
            "upper": row.get("upper_bound"),
            "predicted_for": str(row["predicted_for"])[:10],
        }
    return summary


def evaluate_past_predictions() -> int:
    """
    For every prediction whose date has now passed, look up the actual value
    in crypto_prices, compute accuracy metrics, and store in prediction_outcomes.
    Returns the number of new outcomes recorded.
    """
    pending = db.get_unevaluated_predictions()
    if pending.empty:
        return 0

    evaluated = 0
    # Load price history once per symbol to avoid repeated queries
    history_cache = {}

    for _, row in pending.iterrows():
        sym    = row["symbol"]
        target = row["target"]
        horizon = int(row["horizon_days"])
        pred_date = pd.Timestamp(row["predicted_for"]).date()
        pred_val  = float(row["predicted_value"])

        if sym not in history_cache:
            history_cache[sym] = db.get_price_history(sym, days=730)

        hist = history_cache[sym]
        if hist.empty:
            continue

        hist = hist.copy()
        hist["date"] = pd.to_datetime(hist["timestamp"]).dt.date

        col_map = {
            "price_usd":  "price_usd",
            "volume_24h": "volume_24h",
            "market_cap": "market_cap",
        }
        col = col_map.get(target)
        if not col:
            continue

        # Actual value on the prediction date
        actual_row = hist[hist["date"] == pred_date]
        if actual_row.empty or actual_row.iloc[0][col] is None:
            continue
        actual_val = float(actual_row.iloc[0][col])
        if actual_val == 0:
            continue

        abs_err = abs(pred_val - actual_val)
        pct_err = (pred_val - actual_val) / actual_val * 100

        # Direction: was the prediction trending the right way vs the base price?
        from datetime import timedelta
        base_date = pred_date - timedelta(days=horizon)
        base_row  = hist[hist["date"] == base_date]
        if base_row.empty:
            # Use the nearest earlier date
            earlier = hist[hist["date"] < pred_date]
            base_row = earlier.tail(horizon + 1).head(1)

        if not base_row.empty and base_row.iloc[0][col] is not None:
            base_val = float(base_row.iloc[0][col])
            if base_val > 0:
                direction_correct = (pred_val > base_val) == (actual_val > base_val)
            else:
                direction_correct = False
        else:
            direction_correct = False

        try:
            db.save_prediction_outcome(
                symbol=sym,
                target=target,
                horizon=horizon,
                predicted_for=pred_date,
                predicted_value=pred_val,
                actual_value=actual_val,
                abs_error=abs_err,
                pct_error=pct_err,
                direction_correct=direction_correct,
            )
            evaluated += 1
        except Exception as e:
            print(f"[EVAL] Could not save outcome {sym}/{target}/{horizon}d/{pred_date}: {e}")

    if evaluated:
        print(f"[EVAL] Recorded {evaluated} new prediction outcomes.")
    return evaluated


def get_accuracy_summary() -> dict:
    """
    Return a nested dict of model accuracy stats:
    { symbol -> { horizon_str -> { target -> {directional_accuracy, mean_abs_pct_error, sample_count} } } }
    """
    df = db.get_prediction_accuracy()
    summary = {}
    for _, row in df.iterrows():
        sym  = row["symbol"]
        hor  = str(int(row["horizon_days"]))
        tgt  = row["target"]
        summary.setdefault(sym, {}).setdefault(hor, {})[tgt] = {
            "directional_accuracy": round(float(row["directional_accuracy"]), 1),
            "mean_abs_pct_error":   round(float(row["mean_abs_pct_error"]), 2),
            "sample_count":         int(row["sample_count"]),
        }
    return summary


def get_recommendations() -> dict:
    """
    Rank tracked coins for:
      - daily_top5:   best 24-hour buys (1d ML prediction + 7d momentum)
      - monthly_top5: best 1-month holds (7d ML prediction extrapolated × 4.3 + 30d momentum)

    Returns {daily_top5: [...], monthly_top5: [...], generated_at: str}
    """
    preds = get_prediction_summary()

    candidates = []
    for sym, info in CRYPTO_ASSETS.items():
        try:
            hist = db.get_price_history(sym, days=45)
            if hist.empty or len(hist) < 10:
                continue

            hist = hist.sort_values("timestamp").copy()
            hist["price"] = hist["price_usd"].ffill()

            current_price = float(hist["price"].iloc[-1])
            if current_price <= 0:
                continue

            p7  = float(hist["price"].iloc[-8])  if len(hist) >= 8  else float(hist["price"].iloc[0])
            p30 = float(hist["price"].iloc[-31]) if len(hist) >= 31 else float(hist["price"].iloc[0])
            momentum_7d  = ((current_price / p7)  - 1) * 100 if p7  > 0 else 0.0
            momentum_30d = ((current_price / p30) - 1) * 100 if p30 > 0 else 0.0

            pred_1d_val = preds.get(sym, {}).get("1", {}).get("price_usd", {}).get("value")
            pred_7d_val = preds.get(sym, {}).get("7", {}).get("price_usd", {}).get("value")

            pred_1d_pct = ((pred_1d_val / current_price) - 1) * 100 if pred_1d_val else None
            pred_7d_pct = ((pred_7d_val / current_price) - 1) * 100 if pred_7d_val else None
            monthly_pct = pred_7d_pct * 4.3 if pred_7d_pct is not None else None

            # Composite scores
            daily_score   = 0.7 * pred_1d_pct  + 0.3 * momentum_7d  if pred_1d_pct  is not None else None
            monthly_score = 0.6 * monthly_pct  + 0.4 * momentum_30d if monthly_pct  is not None else None

            # Weekly: rank by absolute size of 7d move; boost when momentum agrees
            if pred_7d_pct is not None:
                momentum_agrees = (pred_7d_pct > 0) == (momentum_7d > 0)
                weekly_score = abs(pred_7d_pct) * (1.15 if momentum_agrees else 0.85)
                weekly_direction = "LONG" if pred_7d_pct >= 0 else "SHORT"
            else:
                weekly_score = None
                weekly_direction = None

            candidates.append({
                "symbol": sym,
                "name": info["name"],
                "color": info["color"],
                "current_price": round(current_price, 2),
                "pred_1d_price":  round(pred_1d_val, 2) if pred_1d_val else None,
                "pred_7d_price":  round(pred_7d_val, 2) if pred_7d_val else None,
                "pred_1d_pct":    round(pred_1d_pct,  2) if pred_1d_pct  is not None else None,
                "pred_7d_pct":    round(pred_7d_pct,  2) if pred_7d_pct  is not None else None,
                "monthly_pct":    round(monthly_pct,  2) if monthly_pct  is not None else None,
                "momentum_7d":    round(momentum_7d,  2),
                "momentum_30d":   round(momentum_30d, 2),
                "daily_score":    daily_score,
                "monthly_score":  monthly_score,
                "weekly_score":   weekly_score,
                "weekly_direction": weekly_direction,
            })
        except Exception as e:
            print(f"[RECS] Error scoring {sym}: {e}")

    def _signal(pct):
        if pct is None:   return "Watch"
        if pct >= 5:      return "Strong Buy"
        if pct >= 2:      return "Buy"
        if pct >= 0:      return "Hold"
        return "Caution"

    daily   = sorted([c for c in candidates if c["daily_score"]   is not None],
                     key=lambda x: x["daily_score"],   reverse=True)
    monthly = sorted([c for c in candidates if c["monthly_score"] is not None],
                     key=lambda x: x["monthly_score"], reverse=True)
    weekly  = sorted([c for c in candidates if c["weekly_score"]  is not None],
                     key=lambda x: x["weekly_score"],  reverse=True)

    for i, c in enumerate(daily[:5]):
        c["rank"]   = i + 1
        c["signal"] = _signal(c["pred_1d_pct"])

    for i, c in enumerate(monthly[:5]):
        c["rank"]   = i + 1
        c["signal"] = _signal(c["monthly_pct"])

    for i, c in enumerate(weekly[:5]):
        c["rank"] = i + 1
        # Confidence label based on momentum agreement
        momentum_agrees = (c["pred_7d_pct"] > 0) == (c["momentum_7d"] > 0)
        c["confidence"] = "High" if momentum_agrees and abs(c["pred_7d_pct"]) >= 3 else \
                          "Medium" if momentum_agrees else "Low"

    return {
        "daily_top5":   daily[:5],
        "monthly_top5": monthly[:5],
        "weekly_top5":  weekly[:5],
        "generated_at": datetime.utcnow().isoformat(),
    }


def get_feature_importance(symbol: str, target: str, horizon: int) -> dict:
    """Return top-20 feature importances for a model."""
    meta = _load_model(symbol, target, horizon)
    if meta is None:
        return {}
    model = meta["model"]
    feat_cols = meta["feat_cols"]
    importances = model.feature_importances_
    pairs = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:20]
    return {k: float(v) for k, v in pairs}
