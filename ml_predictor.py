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

MODEL_VERSION = 2   # v2: daily resampling + %-change targets + baseline evaluation

# Lookback windows (in DAYS — data is resampled to one row per day first)
RET_LAGS = [1, 2, 3, 5, 7, 14, 21, 30]
VOL_WINDOWS = [7, 14, 30]
MA_WINDOWS = [7, 14, 30]


# ---------------------------------------------------------------------------
# Step 1 — one row per day
# ---------------------------------------------------------------------------

def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse mixed-frequency rows (hourly CMC snapshots + daily Coinbase /
    CoinGecko rows) into exactly one row per calendar day (UTC), using the
    last observation of each day. Short gaps (<=3 days) are forward-filled so
    that shift(n) always means n DAYS, never n rows.
    """
    if df.empty:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp")
    d["date"] = d["timestamp"].dt.normalize()
    cols = [c for c in ["price_usd", "volume_24h", "market_cap"] if c in d.columns]
    daily = d.groupby("date")[cols].last()
    daily = daily.asfreq("D").ffill(limit=3)
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# Step 2 — scale-free features, %-change targets
# ---------------------------------------------------------------------------

def _features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build scale-free features (returns, volatility, distance from averages)
    so the model generalises to price levels it has never seen."""
    f = pd.DataFrame(index=daily.index)
    price = daily["price_usd"]
    r1 = price.pct_change(1, fill_method=None)

    for k in RET_LAGS:
        f[f"ret_{k}d"] = price.pct_change(k, fill_method=None)
    for w in VOL_WINDOWS:
        f[f"vol_{w}d"] = r1.rolling(w).std()
    for w in MA_WINDOWS:
        f[f"px_vs_ma{w}"] = price / price.rolling(w).mean() - 1
    f["px_vs_high30"] = price / price.rolling(30).max() - 1
    f["px_vs_low30"] = price / price.rolling(30).min() - 1

    if "volume_24h" in daily:
        v = daily["volume_24h"]
        f["volu_chg_1d"] = v.pct_change(1, fill_method=None)
        f["volu_chg_7d"] = v.pct_change(7, fill_method=None)
        f["volu_vs_ma7"] = v / v.rolling(7).mean() - 1
        f["volu_vs_ma30"] = v / v.rolling(30).mean() - 1
    if "market_cap" in daily:
        m = daily["market_cap"]
        f["mcap_chg_7d"] = m.pct_change(7, fill_method=None)
        f["mcap_chg_30d"] = m.pct_change(30, fill_method=None)

    f["day_of_week"] = f.index.dayofweek
    f["month"] = f.index.month
    return f.replace([np.inf, -np.inf], np.nan)


def _build_features(df: pd.DataFrame, target_col: str, horizon: int) -> tuple:
    """
    Returns (X, y, feature_names) where y is the % change of target_col
    `horizon` DAYS ahead.  (None, None, None) if not enough data.
    """
    daily = _to_daily(df)
    if daily.empty or target_col not in daily or daily[target_col].notna().sum() < MIN_TRAINING_ROWS + horizon:
        return None, None, None
    f = _features(daily)
    tgt = daily[target_col]
    f["target"] = tgt.shift(-horizon) / tgt - 1
    f = f.replace([np.inf, -np.inf], np.nan)
    feat_cols = [c for c in f.columns if c != "target"]
    f = f.dropna(subset=["target", "ret_30d", "vol_30d"])
    if len(f) < MIN_TRAINING_ROWS:
        return None, None, None
    X = f[feat_cols].fillna(0).values
    y = f["target"].values
    return X, y, feat_cols


# ---------------------------------------------------------------------------
# Training (Step 3 — honest out-of-sample evaluation vs a "no change" baseline)
# ---------------------------------------------------------------------------

def _new_model():
    # Deliberately conservative: daily crypto returns are mostly noise, so a
    # shallow, heavily regularised model overfits far less. absolute_error
    # predicts the median move and matches the MAE we evaluate on.
    return GradientBoostingRegressor(
        loss="absolute_error",
        n_estimators=150,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.7,
        min_samples_leaf=30,
        random_state=42,
    )


def train_model(symbol: str, target: str, horizon: int) -> dict:
    """
    Train a model that predicts the % change of `target` over `horizon` days.
    Metrics are all OUT-OF-SAMPLE (walk-forward CV) and compared with the
    naive "no change" forecast:
        mae           – model mean abs error of the % change (e.g. 0.031 = 3.1%)
        baseline_mae  – same for always predicting 0% change
        skill_score   – 1 - mae/baseline_mae  (>0 = beats no-change)
        directional   – % of validation days the predicted direction was right
    """
    print(f"[ML] Training {symbol} | {target} | {horizon}d horizon...")
    df = db.get_price_history(symbol, days=730)
    if df.empty:
        print(f"[ML] No data for {symbol}")
        return {}

    X, y, feat_cols = _build_features(df, target, horizon)
    if X is None:
        print(f"[ML] Not enough daily data for {symbol}/{target}/{horizon}d")
        return {}

    tscv = TimeSeriesSplit(n_splits=4, gap=horizon)   # gap avoids overlap leakage
    preds_all, actual_all = [], []
    for train_idx, val_idx in tscv.split(X):
        gb = _new_model()
        gb.fit(X[train_idx], y[train_idx])
        preds_all.append(gb.predict(X[val_idx]))
        actual_all.append(y[val_idx])
    p = np.concatenate(preds_all)
    a = np.concatenate(actual_all)

    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    baseline_mae = float(np.mean(np.abs(a)))
    skill = float(1 - mae / baseline_mae) if baseline_mae > 0 else 0.0
    nz = a != 0
    directional = float(np.mean(np.sign(p[nz]) == np.sign(a[nz])) * 100) if nz.any() else 50.0
    up_share = float(np.mean(a[nz] > 0) * 100) if nz.any() else 50.0
    oos_r2 = float(r2_score(a, p))

    gb = _new_model()
    gb.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    joblib.dump({
        "version": MODEL_VERSION,
        "model": gb,
        "feat_cols": feat_cols,
        "symbol": symbol,
        "target": target,
        "horizon": horizon,
        "mae_pct": mae,
        "trained_at": datetime.utcnow().isoformat(),
    }, model_path)

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE ml_models SET is_active = 0
            WHERE symbol = ? AND target = ? AND horizon_days = ?
        """, symbol, target, horizon)
        cursor.execute("""
            INSERT INTO ml_models
                (symbol, target, horizon_days, model_file, mae, rmse, r2_score,
                 training_rows, is_active, baseline_mae, skill_score,
                 directional_acc, up_share, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
        """, symbol, target, horizon, model_path, mae, rmse, oos_r2, len(y),
             baseline_mae, skill, directional, up_share, MODEL_VERSION)

    print(f"[ML] {symbol}/{target}/{horizon}d | MAE={mae*100:.2f}% vs no-change {baseline_mae*100:.2f}% "
          f"| skill={skill*100:+.1f}% | dir={directional:.1f}% (up-days {up_share:.0f}%) | rows={len(y)}")
    return {
        "symbol": symbol, "target": target, "horizon": horizon,
        "mae": mae, "baseline_mae": baseline_mae, "skill": skill,
        "directional": directional, "rows": len(y), "model_path": model_path,
    }


def model_is_current(symbol: str, target: str, horizon: int) -> bool:
    """True if a model file exists AND was built by the current MODEL_VERSION."""
    path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    if not os.path.exists(path):
        return False
    try:
        return joblib.load(path).get("version") == MODEL_VERSION
    except Exception:
        return False


def train_all(only_missing: bool = False):
    """Train all models for all symbols, targets, and horizons.
    only_missing=True skips models whose .pkl file already exists."""
    results = []
    for symbol in CRYPTO_ASSETS:
        for target in TARGETS:
            for horizon in PREDICTION_HORIZONS:
                if only_missing and model_is_current(symbol, target, horizon):
                    continue
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
    Predict `target` `horizon` days ahead. The model outputs a % change, which
    is applied to the latest daily value. The confidence band is the model's
    out-of-sample mean absolute % error.
    """
    meta = _load_model(symbol, target, horizon)
    if meta is None or meta.get("version") != MODEL_VERSION:
        return None

    df = db.get_price_history(symbol, days=120)
    daily = _to_daily(df)
    if daily.empty or len(daily) < 35 or target not in daily:
        return None
    current = daily[target].dropna()
    if current.empty:
        return None
    current = float(current.iloc[-1])

    f = _features(daily)
    row = f[meta["feat_cols"]].iloc[-1]
    if row.isnull().all():
        return None
    pct = float(meta["model"].predict(row.fillna(0).values.reshape(1, -1))[0])
    pred_value = current * (1 + pct)
    margin = abs(current) * float(meta.get("mae_pct", 0.05))

    predicted_for = (datetime.utcnow() + timedelta(days=horizon)).replace(
        hour=0, minute=0, second=0, microsecond=0)

    result = {
        "symbol": symbol,
        "target": target,
        "horizon": horizon,
        "predicted_for": predicted_for,
        "predicted_value": pred_value,
        "predicted_pct": pct,
        "lower_bound": max(pred_value - margin, 0),
        "upper_bound": pred_value + margin,
    }
    try:
        db.save_prediction(
            symbol=symbol, target=target, horizon=horizon,
            predicted_for=predicted_for, value=pred_value,
            lower=result["lower_bound"], upper=result["upper_bound"],
        )
    except Exception as e:
        print(f"[ML] Could not save prediction: {e}")
    return result


def get_model_quality() -> dict:
    """
    Out-of-sample backtest stats of the active models:
    { symbol -> { horizon_str -> { target -> {skill, directional, mae_pct, baseline_mae_pct, up_share} } } }
    """
    out = {}
    try:
        with db.get_connection() as conn:
            rows = conn.cursor().execute("""
                SELECT symbol, target, horizon_days, mae, baseline_mae, skill_score,
                       directional_acc, up_share
                FROM ml_models
                WHERE is_active = 1 AND model_version = ?
            """, MODEL_VERSION).fetchall()
        for r in rows:
            out.setdefault(r[0], {}).setdefault(str(int(r[2])), {})[r[1]] = {
                "mae_pct": round(float(r[3]) * 100, 2),
                "baseline_mae_pct": round(float(r[4]) * 100, 2),
                "skill": round(float(r[5]) * 100, 1),
                "directional": round(float(r[6]), 1),
                "up_share": round(float(r[7]), 1),
            }
    except Exception as e:
        print(f"[ML] get_model_quality error: {e}")
    return out


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
    Rank tracked coins for three panels.  Every composite score is multiplied by
    an accuracy weight derived from the model's own historical hit/miss record in
    prediction_outcomes, so models that have been right more often earn a higher
    position and models that have been consistently wrong are discounted.

    accuracy_weight formula (requires ≥5 evaluated outcomes):
        weight = 0.5 + (directional_accuracy_pct / 100)
        → 50 % accuracy  → weight 1.0  (neutral, no boost or penalty)
        → 70 % accuracy  → weight 1.2
        → 30 % accuracy  → weight 0.8
    Below 5 samples the weight stays 1.0 (not enough history to judge).
    """
    preds    = get_prediction_summary()
    accuracy = get_accuracy_summary()   # { sym -> { horizon_str -> { target -> {...} } } }

    def _acc_weight(sym: str, horizon_str: str, min_samples: int = 5) -> tuple:
        """Return (weight, directional_accuracy_pct, sample_count) for price_usd model."""
        stats = accuracy.get(sym, {}).get(horizon_str, {}).get("price_usd", {})
        n     = stats.get("sample_count", 0)
        da    = stats.get("directional_accuracy", 50.0)
        if n < min_samples:
            return 1.0, None, n
        weight = 0.5 + (da / 100.0)   # 50 % → 1.0, 70 % → 1.2, 30 % → 0.8
        return weight, da, n

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

            # Accuracy weights from historical outcomes
            w1d, da_1d, n1d = _acc_weight(sym, "1")
            w7d, da_7d, n7d = _acc_weight(sym, "7")

            # Composite scores — prediction + momentum, scaled by accuracy weight
            daily_score   = (0.7 * pred_1d_pct  + 0.3 * momentum_7d)  * w1d if pred_1d_pct  is not None else None
            monthly_score = (0.6 * monthly_pct  + 0.4 * momentum_30d) * w7d if monthly_pct  is not None else None

            # Weekly: magnitude of 7d move, boosted when momentum agrees, scaled by accuracy
            if pred_7d_pct is not None:
                momentum_agrees = (pred_7d_pct > 0) == (momentum_7d > 0)
                weekly_score = abs(pred_7d_pct) * (1.15 if momentum_agrees else 0.85) * w7d
                weekly_direction = "LONG" if pred_7d_pct >= 0 else "SHORT"
            else:
                weekly_score    = None
                weekly_direction = None
                momentum_agrees  = False

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
                "momentum_agrees":  momentum_agrees,
                # Accuracy metadata shown in UI
                "acc_1d_pct":  round(da_1d, 1) if da_1d is not None else None,
                "acc_7d_pct":  round(da_7d, 1) if da_7d is not None else None,
                "acc_1d_n":    n1d,
                "acc_7d_n":    n7d,
            })
        except Exception as e:
            print(f"[RECS] Error scoring {sym}: {e}")

    def _signal(pct):
        if pct is None:  return "Watch"
        if pct >= 5:     return "Strong Buy"
        if pct >= 2:     return "Buy"
        if pct >= 0:     return "Hold"
        return "Caution"

    def _confidence(c):
        """Derive weekly confidence from accuracy history + momentum agreement."""
        da    = c.get("acc_7d_pct")
        n     = c.get("acc_7d_n", 0)
        agrees = c.get("momentum_agrees", False)
        move  = abs(c.get("pred_7d_pct") or 0)
        # If we have enough history, let accuracy dominate
        if n >= 5 and da is not None:
            if da >= 65 and agrees and move >= 3:  return "High"
            if da >= 55 and agrees:                return "Medium"
            if da < 50:                            return "Low"
            return "Medium"
        # Fall back to momentum-only signal while history builds
        return "High" if agrees and move >= 3 else "Medium" if agrees else "Low"

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
        c["rank"]       = i + 1
        c["confidence"] = _confidence(c)

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
