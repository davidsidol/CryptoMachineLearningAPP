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
try:
    from config import FEATURE_SETS
except ImportError:
    FEATURE_SETS = {}
import time as _time

os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["price_usd", "volume_24h", "market_cap"]

MODEL_VERSION = 3   # v3: + market context inputs (BTC, Fear & Greed, perp funding)

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

# ---------------------------------------------------------------------------
# Market-context inputs (BTC, Fear & Greed, perp funding) — one row per day
# ---------------------------------------------------------------------------

_ctx_cache = {}


def _cached(key, loader, ttl=600):
    hit = _ctx_cache.get(key)
    if hit and _time.time() - hit[0] < ttl:
        return hit[1]
    val = loader()
    _ctx_cache[key] = (_time.time(), val)
    return val


def _enabled_sets(feature_sets=None) -> list:
    fs = FEATURE_SETS if feature_sets is None else feature_sets
    if isinstance(fs, dict):
        return sorted(k for k, v in fs.items() if v)
    return sorted(fs)


def _context(symbol: str, days: int, sets: list) -> pd.DataFrame:
    parts = []
    if "btc" in sets and symbol != "BTC":
        btc = _cached(("btc", days), lambda: _to_daily(db.get_price_history("BTC", days=days)))
        if not btc.empty:
            p = btc["price_usd"]
            r1 = p.pct_change(1, fill_method=None)
            b = pd.DataFrame(index=btc.index)
            for k in (1, 3, 7, 14, 30):
                b[f"btc_ret_{k}d"] = p.pct_change(k, fill_method=None)
            b["btc_vol_14d"] = r1.rolling(14).std()
            b["btc_px_vs_ma30"] = p / p.rolling(30).mean() - 1
            parts.append(b)
    if "fng" in sets:
        fg = _cached(("fng", days), lambda: db.get_fear_greed(days=days + 40))
        if fg is not None and not fg.empty:
            g = fg.copy()
            g["date"] = pd.to_datetime(g["date"])
            v = g.set_index("date").sort_index()["value"].astype(float).asfreq("D").ffill(limit=3)
            parts.append(pd.DataFrame({
                "fng": v,
                "fng_chg_7d": v - v.shift(7),
                "fng_vs_ma30": v - v.rolling(30).mean(),
            }))
    if "funding" in sets:
        fd = _cached(("funding", symbol, days), lambda: db.get_funding_daily(symbol, days=days + 40))
        if fd is not None and not fd.empty:
            d = fd.copy()
            d["date"] = pd.to_datetime(d["date"])
            d = d.set_index("date").sort_index().asfreq("D")
            f1 = d["funding_avg"] * 24 * 365          # annualised rate
            parts.append(pd.DataFrame({
                "fund_1d": f1,
                "fund_3d": f1.rolling(3, min_periods=1).mean(),
                "fund_7d": f1.rolling(7, min_periods=1).mean(),
                "fund_z30": (f1 - f1.rolling(30, min_periods=10).mean()) / f1.rolling(30, min_periods=10).std(),
                "prem_1d": d["premium_avg"],
            }))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)


def _features(daily: pd.DataFrame, ctx: pd.DataFrame = None) -> pd.DataFrame:
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

    if ctx is not None and not ctx.empty:
        c = ctx.reindex(f.index).ffill(limit=3)
        f = f.join(c)
        if "btc_ret_7d" in f:
            f["rel_ret_7d"] = f["ret_7d"] - f["btc_ret_7d"]
            f["rel_ret_30d"] = f["ret_30d"] - f["btc_ret_30d"]
    return f.replace([np.inf, -np.inf], np.nan)


def _build_features(df: pd.DataFrame, target_col: str, horizon: int,
                    ctx: pd.DataFrame = None) -> tuple:
    """
    Returns (X, y, feature_names) where y is the % change of target_col
    `horizon` DAYS ahead.  (None, None, None) if not enough data.
    """
    daily = _to_daily(df)
    if daily.empty or target_col not in daily or daily[target_col].notna().sum() < MIN_TRAINING_ROWS + horizon:
        return None, None, None
    f = _features(daily, ctx)
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


def _cv_eval(X, y, horizon: int) -> dict:
    """Walk-forward CV (gap=horizon) — every number here is out-of-sample."""
    tscv = TimeSeriesSplit(n_splits=4, gap=horizon)
    P, A = [], []
    for tr, va in tscv.split(X):
        m = _new_model()
        m.fit(X[tr], y[tr])
        P.append(m.predict(X[va]))
        A.append(y[va])
    p, a = np.concatenate(P), np.concatenate(A)
    mae = float(mean_absolute_error(a, p))
    base = float(np.mean(np.abs(a)))
    nz = a != 0
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "baseline_mae": base,
        "skill": float(1 - mae / base) if base > 0 else 0.0,
        "directional": float(np.mean(np.sign(p[nz]) == np.sign(a[nz])) * 100) if nz.any() else 50.0,
        "up_share": float(np.mean(a[nz] > 0) * 100) if nz.any() else 50.0,
        "r2": float(r2_score(a, p)),
    }


def train_model(symbol: str, target: str, horizon: int, feature_sets=None) -> dict:
    """
    Train a model that predicts the % change of `target` over `horizon` days,
    using the coin's own history plus the enabled market-context inputs.
    Metrics are out-of-sample and compared with the "no change" forecast.
    """
    sets = _enabled_sets(feature_sets)
    print(f"[ML] Training {symbol} | {target} | {horizon}d | inputs={sets or ['base']}...")
    df = db.get_price_history(symbol, days=730)
    if df.empty:
        print(f"[ML] No data for {symbol}")
        return {}

    ctx = _context(symbol, 730, sets)
    X, y, feat_cols = _build_features(df, target, horizon, ctx)
    if X is None:
        print(f"[ML] Not enough daily data for {symbol}/{target}/{horizon}d")
        return {}

    ev = _cv_eval(X, y, horizon)
    gb = _new_model()
    gb.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    joblib.dump({
        "version": MODEL_VERSION,
        "model": gb,
        "feat_cols": feat_cols,
        "feature_sets": sets,
        "symbol": symbol,
        "target": target,
        "horizon": horizon,
        "mae_pct": ev["mae"],
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
        """, symbol, target, horizon, model_path, ev["mae"], ev["rmse"], ev["r2"], len(y),
             ev["baseline_mae"], ev["skill"], ev["directional"], ev["up_share"], MODEL_VERSION)

    print(f"[ML] {symbol}/{target}/{horizon}d | MAE={ev['mae']*100:.2f}% vs no-change "
          f"{ev['baseline_mae']*100:.2f}% | skill={ev['skill']*100:+.1f}% | dir={ev['directional']:.1f}%")
    return {"symbol": symbol, "target": target, "horizon": horizon, "rows": len(y),
            "model_path": model_path, **ev}


ABLATION_CONFIGS = {
    "base":     [],
    "+btc":     ["btc"],
    "+fng":     ["fng"],
    "+funding": ["funding"],
    "all":      ["btc", "fng", "funding"],
}


def ablation_report(target: str = "price_usd") -> pd.DataFrame:
    """Out-of-sample skill/direction for each input set, per coin and horizon.
    Nothing is saved — this only measures which inputs help."""
    rows = []
    for symbol in CRYPTO_ASSETS:
        df = db.get_price_history(symbol, days=730)
        if df.empty:
            continue
        for name, sets in ABLATION_CONFIGS.items():
            ctx = _context(symbol, 730, sets)
            for h in PREDICTION_HORIZONS:
                X, y, _ = _build_features(df, target, h, ctx)
                if X is None:
                    continue
                ev = _cv_eval(X, y, h)
                rows.append({"symbol": symbol, "horizon": h, "inputs": name,
                             "skill": round(ev["skill"] * 100, 2),
                             "directional": round(ev["directional"], 1),
                             "mae_pct": round(ev["mae"] * 100, 2),
                             "baseline_pct": round(ev["baseline_mae"] * 100, 2)})
    return pd.DataFrame(rows)


# ===========================================================================
# Up/down (direction) classifiers — experimental, run alongside the
# %-change regressors. Nothing here is used by the live app until enabled.
# ===========================================================================

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CLASSIFIER_VERSION = 1


def _new_classifier(kind: str = "gb"):
    if kind == "logit":
        # Strongly regularised linear model: hard to overfit noise
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    return GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=2,
        subsample=0.7, min_samples_leaf=30, random_state=42,
    )


def _class_data(df: pd.DataFrame, horizon: int, ctx: pd.DataFrame = None):
    """X, y_pct (future % change of price), feature names. Rows with no move are dropped."""
    X, y, feat_cols = _build_features(df, "price_usd", horizon, ctx)
    if X is None:
        return None, None, None
    keep = y != 0
    return X[keep], y[keep], feat_cols


def _cv_eval_class(X, y_pct, horizon: int, feat_cols: list, kind: str = "gb") -> dict:
    """
    Walk-forward CV for an up/down classifier, compared with three simple rules
    that need no model at all:
      majority  – always predict whichever direction was more common in training
      trend     – predict the same direction as the last `horizon` days' move
      always_up – always predict up
    Also reports probability quality (Brier score vs. the training up-rate) and
    accuracy when the model is at least 55/45 confident.
    """
    y = (y_pct > 0).astype(int)
    trend_col = f"ret_{horizon}d" if f"ret_{horizon}d" in feat_cols else "ret_1d"
    ti = feat_cols.index(trend_col)
    tscv = TimeSeriesSplit(n_splits=4, gap=horizon)
    P, Y, MAJ, TR, CLIM = [], [], [], [], []
    for tr, va in tscv.split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        m = _new_classifier(kind)
        m.fit(X[tr], y[tr])
        P.append(m.predict_proba(X[va])[:, 1])
        Y.append(y[va])
        up_rate = y[tr].mean()
        MAJ.append(np.full(len(va), 1 if up_rate >= 0.5 else 0))
        TR.append((X[va, ti] > 0).astype(int))
        CLIM.append(np.full(len(va), up_rate))
    p, yv = np.concatenate(P), np.concatenate(Y)
    maj, trd, clim = np.concatenate(MAJ), np.concatenate(TR), np.concatenate(CLIM)
    pred = (p >= 0.5).astype(int)
    conf = np.abs(p - 0.5) >= 0.05
    brier = float(np.mean((p - yv) ** 2))
    brier_clim = float(np.mean((clim - yv) ** 2))
    return {
        "n": int(len(yv)),
        "acc": float((pred == yv).mean() * 100),
        "acc_majority": float((maj == yv).mean() * 100),
        "acc_trend": float((trd == yv).mean() * 100),
        "acc_always_up": float(yv.mean() * 100),
        "best_simple": float(max((maj == yv).mean(), (trd == yv).mean(), yv.mean(), 1 - yv.mean()) * 100),
        "brier": brier,
        "brier_skill": float((1 - brier / brier_clim) * 100) if brier_clim > 0 else 0.0,
        "conf_coverage": float(conf.mean() * 100),
        "conf_acc": float((pred[conf] == yv[conf]).mean() * 100) if conf.any() else float("nan"),
        "pred_up_share": float(pred.mean() * 100),
    }


def classifier_backtest(kinds=("gb", "logit"), input_sets=None) -> pd.DataFrame:
    """Backtest up/down classifiers for every coin and horizon. Saves nothing."""
    if input_sets is None:
        input_sets = {"base": [], "all": ["btc", "fng", "funding"]}
    rows = []
    for symbol in CRYPTO_ASSETS:
        df = db.get_price_history(symbol, days=730)
        if df.empty:
            continue
        for set_name, sets in input_sets.items():
            ctx = _context(symbol, 730, sets)
            for h in PREDICTION_HORIZONS:
                X, y, fc = _class_data(df, h, ctx)
                if X is None:
                    continue
                for kind in kinds:
                    ev = _cv_eval_class(X, y, h, fc, kind)
                    rows.append({"symbol": symbol, "horizon": h, "inputs": set_name, "model": kind, **ev})
    return pd.DataFrame(rows)


def train_classifier(symbol: str, horizon: int, kind: str = "logit", feature_sets=None) -> dict:
    """Fit and save an up/down classifier (models/<SYM>_dir_<h>d.pkl). Not used live yet."""
    sets = _enabled_sets(feature_sets)
    df = db.get_price_history(symbol, days=730)
    ctx = _context(symbol, 730, sets)
    X, y, fc = _class_data(df, horizon, ctx)
    if X is None:
        return {}
    ev = _cv_eval_class(X, y, horizon, fc, kind)
    m = _new_classifier(kind)
    m.fit(X, (y > 0).astype(int))
    path = os.path.join(MODEL_DIR, f"{symbol}_dir_{horizon}d.pkl")
    joblib.dump({"version": CLASSIFIER_VERSION, "kind": kind, "model": m, "feat_cols": fc,
                 "feature_sets": sets, "symbol": symbol, "horizon": horizon, "cv": ev,
                 "trained_at": datetime.utcnow().isoformat()}, path)
    return {"symbol": symbol, "horizon": horizon, "path": path, **ev}


# ===========================================================================
# Pooled full-history PRICE model (v4)
#   One model per horizon, trained on every coin's clean daily candles
#   (single UTC close per day, back to 2017 where available). Targets are
#   volatility-scaled so coins of different temperament can share a model.
#   Replaces the 27 per-coin price models; volume / market-cap stay per-coin.
# ===========================================================================

from sklearn.ensemble import HistGradientBoostingRegressor

POOLED_VERSION = 4
POOLED_TEST_DAYS = 420          # walk-forward evaluation window
PER_COIN_TARGETS = [t for t in TARGETS if t != "price_usd"]


def _pooled_path(horizon: int) -> str:
    return os.path.join(MODEL_DIR, f"POOLED_price_usd_{horizon}d.pkl")


def _new_pooled_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=3, learning_rate=0.03, max_iter=200,
        min_samples_leaf=40, l2_regularization=1.0, random_state=42,
    )


def _pooled_frame() -> pd.DataFrame:
    """Feature table for all coins from daily_candles. Includes the latest day
    (targets NaN) so the same table serves training and prediction."""
    D = db.get_daily_candles()
    if D is None or D.empty or "BTC" not in set(D["symbol"]):
        return pd.DataFrame()
    D = D.copy()
    D["date"] = pd.to_datetime(D["date"])
    px = D.pivot(index="date", columns="symbol", values="close").asfreq("D").ffill(limit=3)
    vol = D.pivot(index="date", columns="symbol", values="volume_usd").asfreq("D")
    hi = D.pivot(index="date", columns="symbol", values="high").asfreq("D")
    lo = D.pivot(index="date", columns="symbol", values="low").asfreq("D")

    try:
        fg = db.get_fear_greed(days=6000)
        fg["date"] = pd.to_datetime(fg["date"])
        fng = fg.set_index("date").sort_index()["value"].astype(float)
    except Exception:
        fng = pd.Series(dtype=float)
    try:
        fu_all = db.get_all_funding_daily()
        fu_all["date"] = pd.to_datetime(fu_all["date"])
    except Exception:
        fu_all = pd.DataFrame(columns=["symbol", "date", "funding_avg"])

    b = px["BTC"]
    br = b.pct_change(fill_method=None)
    frames = []
    for s in px.columns:
        p = px[s]
        r1 = p.pct_change(fill_method=None)
        f = pd.DataFrame(index=px.index)
        for k in RET_LAGS:
            f[f"ret_{k}d"] = p.pct_change(k, fill_method=None)
        for w in (7, 14, 30):
            f[f"vol_{w}d"] = r1.rolling(w).std()
            f[f"px_vs_ma{w}"] = p / p.rolling(w).mean() - 1
        f["px_vs_high30"] = p / p.rolling(30).max() - 1
        f["px_vs_low30"] = p / p.rolling(30).min() - 1
        f["px_vs_high90"] = p / p.rolling(90).max() - 1
        f["vol_ratio"] = f["vol_7d"] / f["vol_30d"]
        f["range_7d"] = ((hi[s] - lo[s]) / p).rolling(7).mean()
        v = vol[s].replace(0, np.nan)
        f["volu_chg_1d"] = v.pct_change(1, fill_method=None).clip(-5, 5)
        f["volu_vs_ma7"] = (v / v.rolling(7).mean() - 1).clip(-5, 5)
        f["volu_vs_ma30"] = (v / v.rolling(30).mean() - 1).clip(-5, 5)
        f["dow"] = f.index.dayofweek
        for k in (1, 3, 7, 14, 30):
            f[f"btc_ret_{k}d"] = b.pct_change(k, fill_method=None)
        f["btc_vol_14d"] = br.rolling(14).std()
        f["btc_px_vs_ma30"] = b / b.rolling(30).mean() - 1
        f["rel_ret_7d"] = f["ret_7d"] - f["btc_ret_7d"]
        f["rel_ret_30d"] = f["ret_30d"] - f["btc_ret_30d"]
        f["beta_60"] = r1.rolling(60).cov(br) / br.rolling(60).var()
        g = fng.reindex(f.index).ffill(limit=3) if len(fng) else pd.Series(np.nan, index=f.index)
        f["fng"] = g
        f["fng_chg_7d"] = g - g.shift(7)
        f["fng_vs_ma30"] = g - g.rolling(30).mean()
        fu = fu_all[fu_all["symbol"] == s].drop_duplicates("date").set_index("date")["funding_avg"]
        fu = fu.reindex(f.index) * 24 * 365
        f["fund_1d"] = fu
        f["fund_7d"] = fu.rolling(7, min_periods=1).mean()
        f["fund_z30"] = (fu - fu.rolling(30, min_periods=10).mean()) / fu.rolling(30, min_periods=10).std()
        f["symbol"] = s
        f["close"] = p
        f["vol30"] = f["vol_30d"]
        for h in PREDICTION_HORIZONS:
            f[f"y{h}"] = p.shift(-h) / p - 1
        frames.append(f.reset_index())
    A = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=["ret_30d", "vol_30d", "close"])
    A = A[A["vol30"] > 0]
    for c in ("ret_7d", "ret_30d", "vol_30d", "volu_vs_ma30", "rel_ret_7d"):
        A["xs_" + c] = A.groupby("date")[c].rank(pct=True)
    return A


def _pooled_feature_cols(A: pd.DataFrame) -> list:
    meta = {"date", "symbol", "close", "vol30"} | {f"y{h}" for h in PREDICTION_HORIZONS}
    return [c for c in A.columns if c not in meta]


def _pooled_walk_forward(A: pd.DataFrame, h: int, cols: list) -> pd.DataFrame:
    """Monthly walk-forward over the last POOLED_TEST_DAYS: every prediction is
    made by a model that never saw that month (with an h-day gap)."""
    ycol = f"y{h}"
    last = A["date"].max()
    fs = last - pd.Timedelta(days=POOLED_TEST_DAYS)
    out = []
    while fs < last:
        fe = fs + pd.Timedelta(days=30)
        tr = A[(A["date"] < fs - pd.Timedelta(days=h)) & A[ycol].notna()]
        te = A[(A["date"] >= fs) & (A["date"] < fe) & A[ycol].notna()]
        if len(tr) > 500 and not te.empty:
            use = [c for c in cols if tr[c].notna().sum() >= 20]
            m = _new_pooled_model().fit(tr[use].values, (tr[ycol] / (tr["vol30"] * np.sqrt(h))).values)
            o = te[["date", "symbol"]].copy()
            o["pred"] = m.predict(te[use].values) * (te["vol30"].values * np.sqrt(h))
            o["act"] = te[ycol].values
            out.append(o)
        fs = fe
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "symbol", "pred", "act"])


def _dir_stats(x: pd.DataFrame) -> dict:
    x = x[x["act"] != 0]
    if x.empty:
        return {}
    mae = float((x["pred"] - x["act"]).abs().mean())
    base = float(x["act"].abs().mean())
    return {
        "n": int(len(x)), "mae": mae, "baseline_mae": base,
        "rmse": float(np.sqrt(((x["pred"] - x["act"]) ** 2).mean())),
        "skill": float(1 - mae / base) if base > 0 else 0.0,
        "directional": float((np.sign(x["pred"]) == np.sign(x["act"])).mean() * 100),
        "up_share": float((x["act"] > 0).mean() * 100),
    }


def train_pooled_all() -> list:
    """Train the pooled price model for every horizon, record honest walk-forward
    metrics per coin in ml_models, and write logs/pooled_report.txt."""
    A = _pooled_frame()
    if A.empty:
        print("[POOLED] No daily candles yet — cannot train.")
        return []
    cols = _pooled_feature_cols(A)
    results, lines = [], []
    lines.append(f"Pooled price model v{POOLED_VERSION} — trained {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    lines.append(f"Daily candles: {len(A)} rows, {A['symbol'].nunique()} coins, "
                 f"{A['date'].min():%Y-%m-%d} to {A['date'].max():%Y-%m-%d}")
    for h in PREDICTION_HORIZONS:
        ycol = f"y{h}"
        print(f"[POOLED] Training price model, {h}d horizon...")
        wf = _pooled_walk_forward(A, h, cols)
        overall = _dir_stats(wf)
        per_sym = {s: _dir_stats(g) for s, g in wf.groupby("symbol")}
        tr = A[A[ycol].notna()]
        use = [c for c in cols if tr[c].notna().sum() >= 20]
        model = _new_pooled_model().fit(tr[use].values, (tr[ycol] / (tr["vol30"] * np.sqrt(h))).values)
        joblib.dump({
            "version": POOLED_VERSION, "model": model, "feat_cols": use, "horizon": h,
            "overall": overall, "per_symbol": per_sym, "rows": int(len(tr)),
            "trained_at": datetime.utcnow().isoformat(),
        }, _pooled_path(h))
        try:
            with db.get_connection() as conn:
                cur = conn.cursor()
                for sym in CRYPTO_ASSETS:
                    st = per_sym.get(sym)
                    if not st:
                        continue
                    cur.execute("""UPDATE ml_models SET is_active = 0
                                   WHERE symbol = ? AND target = 'price_usd' AND horizon_days = ?""", sym, h)
                    cur.execute("""
                        INSERT INTO ml_models
                            (symbol, target, horizon_days, model_file, mae, rmse, r2_score,
                             training_rows, is_active, baseline_mae, skill_score,
                             directional_acc, up_share, model_version)
                        VALUES (?, 'price_usd', ?, ?, ?, ?, 0, ?, 1, ?, ?, ?, ?, ?)
                    """, sym, h, _pooled_path(h), st["mae"], st["rmse"], int(len(tr)),
                         st["baseline_mae"], st["skill"], st["directional"], st["up_share"], POOLED_VERSION)
        except Exception as e:
            print(f"[POOLED] Could not record metrics: {e}")
        if overall:
            msg = (f"{h}d: direction {overall['directional']:.1f}% | error {overall['mae']*100:.2f}% vs "
                   f"no-change {overall['baseline_mae']*100:.2f}% (skill {overall['skill']*100:+.1f}%) | "
                   f"test n={overall['n']} | trained on {len(tr)} rows")
            print(f"[POOLED] {msg}")
            lines.append(msg)
            lines.append("   per coin dir%: " + ", ".join(
                f"{s} {per_sym[s]['directional']:.0f}" for s in CRYPTO_ASSETS if per_sym.get(s)))
        results.append({"horizon": h, **overall})
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "pooled_report.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[POOLED] Could not write report: {e}")
    _ctx_cache.pop("pooled_frame", None)
    return results


def pooled_is_current(horizon: int) -> bool:
    try:
        return joblib.load(_pooled_path(horizon)).get("version") == POOLED_VERSION
    except Exception:
        return False


def pooled_age_days() -> float:
    try:
        t = datetime.fromisoformat(joblib.load(_pooled_path(PREDICTION_HORIZONS[0]))["trained_at"])
        return (datetime.utcnow() - t).total_seconds() / 86400
    except Exception:
        return 1e9


def retrain_if_stale(max_age_days: float = 7) -> bool:
    """Weekly retrain of the pooled price model and the per-coin volume / market-cap models."""
    if pooled_age_days() < max_age_days:
        return False
    print("[RETRAIN] Models are older than a week — retraining...")
    train_pooled_all()
    train_all(only_missing=False)
    return True


def _predict_price_pooled(symbol: str, horizon: int) -> dict:
    """Forecast the daily CLOSE `horizon` days after the last completed UTC day."""
    try:
        meta = joblib.load(_pooled_path(horizon))
    except Exception:
        return None
    if meta.get("version") != POOLED_VERSION:
        return None
    A = _cached("pooled_frame", _pooled_frame, ttl=900)
    if A is None or A.empty:
        return None
    rows = A[A["symbol"] == symbol]
    if rows.empty:
        return None
    row = rows.sort_values("date").iloc[-1]
    last_day = pd.Timestamp(row["date"]).date()
    if last_day < (datetime.utcnow().date() - timedelta(days=1)):
        _ctx_cache.pop("pooled_frame", None)     # stale candles: try fresh next time
        return None
    x = row.reindex(meta["feat_cols"]).astype(float).values.reshape(1, -1)
    pct = float(float(meta["model"].predict(x)[0]) * float(row["vol30"]) * np.sqrt(horizon))
    base = float(row["close"])
    pred_value = float(base * (1 + pct))
    st = (meta.get("per_symbol") or {}).get(symbol) or meta.get("overall") or {}
    margin = float(base * float(st.get("mae", 0.05)))
    predicted_for = datetime.combine(last_day + timedelta(days=horizon), datetime.min.time())
    result = {
        "symbol": symbol, "target": "price_usd", "horizon": horizon,
        "predicted_for": predicted_for, "predicted_value": pred_value, "predicted_pct": pct,
        "lower_bound": max(pred_value - margin, 0), "upper_bound": pred_value + margin,
    }
    try:
        db.save_prediction(symbol=symbol, target="price_usd", horizon=horizon,
                           predicted_for=predicted_for, value=pred_value,
                           lower=result["lower_bound"], upper=result["upper_bound"])
    except Exception as e:
        print(f"[POOLED] Could not save prediction: {e}")
    return result


def _daily_close_lookup() -> dict:
    def load():
        D = db.get_daily_candles()
        if D is None or D.empty:
            return {}
        return {(s, pd.Timestamp(d).date()): float(c) for s, d, c in zip(D["symbol"], D["date"], D["close"])}
    return _cached("daily_close_lookup", load, ttl=900)


# Hidden live test: 1-day linear up/down model with all inputs (best backtest cell)
SHADOW_HORIZON = 1
SHADOW_KIND = "logit"
SHADOW_SETS = ["btc", "fng", "funding"]


def _shadow_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_dir_{SHADOW_HORIZON}d.pkl")


def ensure_shadow_models():
    """Train the hidden up/down models once (kept fixed for the test period)."""
    for symbol in CRYPTO_ASSETS:
        try:
            path = _shadow_path(symbol)
            if os.path.exists(path):
                meta = joblib.load(path)
                if meta.get("version") == CLASSIFIER_VERSION and meta.get("kind") == SHADOW_KIND:
                    continue
            r = train_classifier(symbol, SHADOW_HORIZON, SHADOW_KIND, SHADOW_SETS)
            if r:
                print(f"[SHADOW] Trained {symbol} up/down model (backtest acc {r['acc']:.1f}%)")
        except Exception as e:
            print(f"[SHADOW] Train error {symbol}: {e}")


def shadow_predict_all() -> int:
    """Log an up/down probability per coin for tomorrow's close. Not displayed."""
    n = 0
    for symbol in CRYPTO_ASSETS:
        try:
            path = _shadow_path(symbol)
            if not os.path.exists(path):
                continue
            meta = joblib.load(path)
            daily = _to_daily(db.get_price_history(symbol, days=120))
            if daily.empty or len(daily) < 35:
                continue
            ctx = _context(symbol, 120, meta.get("feature_sets", []))
            row = _features(daily, ctx).reindex(columns=meta["feat_cols"]).iloc[-1]
            prob = float(meta["model"].predict_proba(row.fillna(0).values.reshape(1, -1))[0, 1])
            base = float(daily["price_usd"].dropna().iloc[-1])
            target = (datetime.utcnow() + timedelta(days=SHADOW_HORIZON)).date()
            db.insert_shadow(symbol, SHADOW_HORIZON, target, base, prob)
            n += 1
        except Exception as e:
            print(f"[SHADOW] Predict error {symbol}: {e}")
    return n


def score_shadow() -> int:
    """Score logged up/down calls once their target day has closed."""
    try:
        pending = db.get_unscored_shadow()
    except Exception as e:
        print(f"[SHADOW] Read error: {e}")
        return 0
    if pending.empty:
        return 0
    n, cache = 0, {}
    for _, r in pending.iterrows():
        sym = r["symbol"]
        if sym not in cache:
            h = db.get_price_history(sym, days=60)
            h["date"] = pd.to_datetime(h["timestamp"]).dt.date
            cache[sym] = h
        day = cache[sym][cache[sym]["date"] == pd.Timestamp(r["target_date"]).date()].sort_values("timestamp")
        if day.empty:
            continue
        actual = float(day["price_usd"].iloc[-1])
        base = float(r["base_price"])
        if actual == base:
            continue
        correct = (float(r["prob_up"]) >= 0.5) == (actual > base)
        try:
            db.score_shadow(int(r["id"]), actual, bool(correct))
            n += 1
        except Exception as e:
            print(f"[SHADOW] Score error: {e}")
    if n:
        print(f"[SHADOW] Scored {n} up/down calls.")
    return n


def model_is_current(symbol: str, target: str, horizon: int) -> bool:
    """True if a model file exists AND was built by the current MODEL_VERSION."""
    path = os.path.join(MODEL_DIR, f"{symbol}_{target}_{horizon}d.pkl")
    if not os.path.exists(path):
        return False
    try:
        meta = joblib.load(path)
        return (meta.get("version") == MODEL_VERSION
                and meta.get("feature_sets", []) == _enabled_sets())
    except Exception:
        return False


def train_all(only_missing: bool = False):
    """Train all models for all symbols, targets, and horizons.
    only_missing=True skips models whose .pkl file already exists."""
    results = []
    for symbol in CRYPTO_ASSETS:
        for target in PER_COIN_TARGETS:      # price_usd is handled by the pooled model
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
    if target == "price_usd":
        return _predict_price_pooled(symbol, horizon)

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

    ctx = _context(symbol, 120, meta.get("feature_sets", []))
    f = _features(daily, ctx).reindex(columns=meta["feat_cols"])
    row = f.iloc[-1]
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
                WHERE is_active = 1 AND model_version >= ?
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

        # Pooled price forecasts are made from the last completed daily close and
        # target the daily close `horizon` days later — score them on the same
        # clean daily candles. (Recognised by target_day - horizon < generation day.)
        gen_day = pd.Timestamp(row["generated_at"]).date() if "generated_at" in row and pd.notna(row["generated_at"]) else None
        if target == "price_usd" and gen_day is not None and (pred_date - timedelta(days=horizon)) < gen_day:
            closes = _daily_close_lookup()
            actual_val = closes.get((sym, pred_date))
            base_val = closes.get((sym, pred_date - timedelta(days=horizon)))
            if not actual_val or not base_val:
                continue
            try:
                db.save_prediction_outcome(
                    symbol=sym, target=target, horizon=horizon, predicted_for=pred_date,
                    predicted_value=pred_val, actual_value=actual_val,
                    abs_error=abs(pred_val - actual_val),
                    pct_error=(pred_val - actual_val) / actual_val * 100,
                    direction_correct=(pred_val > base_val) == (actual_val > base_val),
                )
                evaluated += 1
            except Exception as e:
                print(f"[EVAL] Could not save outcome {sym}/{target}/{horizon}d/{pred_date}: {e}")
            continue

        # Forecasts target the CLOSE of the predicted day (that is what the models
        # are trained on), so use the last observation of that day...
        day_rows = hist[hist["date"] == pred_date].sort_values("timestamp")
        if day_rows.empty or pd.isna(day_rows.iloc[-1][col]):
            continue
        actual_val = float(day_rows.iloc[-1][col])
        if actual_val == 0:
            continue

        abs_err = abs(pred_val - actual_val)
        pct_err = (pred_val - actual_val) / actual_val * 100

        # ...and judge direction from the value at the moment the forecast was made.
        gen_at = pd.Timestamp(row["generated_at"]) if "generated_at" in row and pd.notna(row["generated_at"]) else None
        ts = pd.to_datetime(hist["timestamp"])
        if gen_at is not None:
            base_rows = hist[ts <= gen_at].sort_values("timestamp")
        else:
            base_rows = hist[hist["date"] < pred_date].sort_values("timestamp").tail(1)
        if not base_rows.empty and pd.notna(base_rows.iloc[-1][col]) and float(base_rows.iloc[-1][col]) > 0:
            base_val = float(base_rows.iloc[-1][col])
            direction_correct = (pred_val > base_val) == (actual_val > base_val)
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
