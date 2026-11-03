"""
Flask web application for Crypto ML Forecasting Dashboard
"""
import threading
import time
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify, request

import database as db
import ml_predictor as ml
import data_collector as dc
from config import (
    FLASK_SECRET_KEY, FLASK_PORT, FLASK_DEBUG,
    CRYPTO_ASSETS, PREDICTION_HORIZONS,
    DATA_REFRESH_MINUTES, PREDICTION_REFRESH_HOURS
)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Background refresh state
_refresh_lock = threading.Lock()
_last_refresh = {"time": None, "status": "never"}


def _data_refresh_loop():
    """Pull new market data every DATA_REFRESH_MINUTES, then score past predictions."""
    while True:
        time.sleep(DATA_REFRESH_MINUTES * 60)
        with _refresh_lock:
            try:
                dc.refresh_all()
                ml.evaluate_past_predictions()
                _last_refresh["time"] = datetime.now(timezone.utc).isoformat()
                _last_refresh["status"] = "ok"
            except Exception as e:
                _last_refresh["status"] = f"error: {e}"


def _prediction_refresh_loop():
    """Regenerate ML predictions every PREDICTION_REFRESH_HOURS."""
    while True:
        time.sleep(PREDICTION_REFRESH_HOURS * 3600)
        try:
            ml.predict_all()
        except Exception as e:
            print(f"[APP] Prediction refresh error: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard."""
    try:
        latest = db.get_latest_prices()
        prices = {}
        for _, row in latest.iterrows():
            prices[row["symbol"]] = {
                "price": row["price_usd"],
                "volume": row["volume_24h"],
                "market_cap": row["market_cap"],
                "pct_24h": row.get("pct_change_24h"),
                "pct_7d": row.get("pct_change_7d"),
                "updated": str(row["timestamp"])[:19],
            }
    except Exception as e:
        print(f"[APP] Error fetching latest prices: {e}")
        prices = {}

    try:
        predictions = ml.get_prediction_summary()
    except Exception as e:
        print(f"[APP] Error fetching predictions: {e}")
        predictions = {}

    try:
        accuracy = ml.get_accuracy_summary()
    except Exception as e:
        print(f"[APP] Error fetching accuracy: {e}")
        accuracy = {}

    try:
        recommendations = ml.get_recommendations()
    except Exception as e:
        print(f"[APP] Error fetching recommendations: {e}")
        recommendations = {"daily_top5": [], "monthly_top5": [], "weekly_top5": [], "generated_at": ""}

    try:
        quality = ml.get_model_quality()
    except Exception as e:
        print(f"[APP] Error fetching model quality: {e}")
        quality = {}

    fear_greed = _fear_greed_summary()
    market = _market_context_summary(prices)

    return render_template(
        "index.html",
        quality=quality,
        fear_greed=fear_greed,
        market=market,
        assets=CRYPTO_ASSETS,
        prices=prices,
        predictions=predictions,
        accuracy=accuracy,
        recommendations=recommendations,
        horizons=PREDICTION_HORIZONS,
        last_refresh=_last_refresh,
    )


def _fear_greed_summary() -> dict:
    """Latest Fear & Greed value plus yesterday / last week / last month."""
    try:
        df = db.get_fear_greed(days=31)
    except Exception as e:
        print(f"[APP] Fear & Greed read error: {e}")
        return {}
    if df.empty:
        return {}
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    def at(i):
        if len(df) > i:
            return {"value": int(df.loc[i, "value"]), "label": df.loc[i, "classification"]}
        return None

    return {
        "value": int(df.loc[0, "value"]),
        "label": df.loc[0, "classification"],
        "date": str(df.loc[0, "date"])[:10],
        "yesterday": at(1),
        "week": at(7),
        "month": at(30),
        "series": [int(v) for v in df["value"][::-1].tolist()],
    }


def _positioning(oi_chg, px_chg):
    """Classic OI-vs-price read of what futures traders are doing."""
    if oi_chg is None or px_chg is None:
        return None
    if abs(oi_chg) < 1.0:
        return ("Flat positioning", "neutral")
    if oi_chg > 0 and px_chg >= 0:
        return ("New longs opening", "long")
    if oi_chg > 0 and px_chg < 0:
        return ("New shorts opening", "short")
    if oi_chg < 0 and px_chg >= 0:
        return ("Shorts covering", "long")
    return ("Longs closing", "short")


def _market_context_summary(prices: dict) -> dict:
    """Perp funding / open interest / leverage per coin and BTC/ETH options."""
    out = {"futures": [], "options": [], "updated": None}
    try:
        fut = db.get_futures_latest()
    except Exception as e:
        print(f"[APP] futures read error: {e}")
        fut = None
    if fut is not None and not fut.empty:
        fut = fut.set_index("symbol")
        for sym in CRYPTO_ASSETS:
            if sym not in fut.index:
                continue
            r = fut.loc[sym]
            apr = float(r["funding"]) * 24 * 365 * 100 if r["funding"] is not None else None
            oi = float(r["oi_usd"]) if r["oi_usd"] is not None else None
            prev = r.get("oi_usd_24h_ago")
            oi_chg = (oi / float(prev) - 1) * 100 if oi and prev and float(prev) > 0 else None
            p = prices.get(sym, {})
            mcap = p.get("market_cap")
            px_chg = p.get("pct_24h")
            bias = None
            if apr is not None:
                bias = ("Longs paying", "long") if apr > 5 else (("Shorts paying", "short") if apr < -1 else ("Balanced", "neutral"))
            out["futures"].append({
                "symbol": sym,
                "name": CRYPTO_ASSETS[sym]["name"],
                "cmc_id": CRYPTO_ASSETS[sym]["cmc_id"],
                "color": CRYPTO_ASSETS[sym]["color"],
                "funding_apr": apr,
                "bias": bias,
                "oi_usd": oi,
                "oi_chg_24h": oi_chg,
                "oi_to_mcap": (oi / mcap * 100) if oi and mcap else None,
                "max_leverage": int(r["max_leverage"]) if r["max_leverage"] else None,
                "positioning": _positioning(oi_chg, px_chg),
                "day_volume_usd": float(r["day_volume_usd"]) if r["day_volume_usd"] else None,
            })
            out["updated"] = str(r["ts"])[:16]
    try:
        opt = db.get_options_latest()
        for _, r in opt.iterrows():
            pcr = float(r["put_call_ratio"]) if r["put_call_ratio"] is not None else None
            dvol = float(r["dvol"]) if r["dvol"] is not None else None
            out["options"].append({
                "currency": r["currency"],
                "cmc_id": CRYPTO_ASSETS.get(r["currency"], {}).get("cmc_id"),
                "put_call_ratio": pcr,
                "lean": None if pcr is None else ("Bullish lean" if pcr < 0.7 else ("Bearish lean" if pcr > 1.0 else "Neutral")),
                "oi_usd": float(r["oi_usd"]) if r["oi_usd"] is not None else None,
                "dvol": dvol,
                "implied_daily_move": dvol / (365 ** 0.5) if dvol else None,
                "call_oi": float(r["call_oi"] or 0),
                "put_oi": float(r["put_oi"] or 0),
            })
    except Exception as e:
        print(f"[APP] options read error: {e}")
    return out


@app.route("/api/market_context")
def api_market_context():
    try:
        latest = db.get_latest_prices()
        prices = {row["symbol"]: {"market_cap": row["market_cap"], "pct_24h": row.get("pct_change_24h")}
                  for _, row in latest.iterrows()}
    except Exception:
        prices = {}
    return jsonify(_market_context_summary(prices))


@app.route("/api/fear_greed")
def api_fear_greed():
    return jsonify(_fear_greed_summary())


@app.route("/api/prices/<symbol>")
def api_prices(symbol):
    """Return historical price data as JSON."""
    symbol = symbol.upper()
    if symbol not in CRYPTO_ASSETS:
        return jsonify({"error": "Unknown symbol"}), 404

    days = int(request.args.get("days", 90))
    try:
        df = db.get_price_history(symbol, days=days)
        if df.empty:
            return jsonify({"symbol": symbol, "data": []})

        df["timestamp"] = df["timestamp"].astype(str)
        return jsonify({
            "symbol": symbol,
            "name": CRYPTO_ASSETS[symbol]["name"],
            "data": df[["timestamp", "price_usd", "volume_24h", "market_cap"]].to_dict("records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ohlcv/<symbol>")
def api_ohlcv(symbol):
    """Return OHLCV candle data as JSON."""
    symbol = symbol.upper()
    if symbol not in CRYPTO_ASSETS:
        return jsonify({"error": "Unknown symbol"}), 404

    days = int(request.args.get("days", 90))
    try:
        df = db.get_ohlcv_history(symbol, days=days)
        if df.empty:
            return jsonify({"symbol": symbol, "data": []})
        df["timestamp"] = df["timestamp"].astype(str)
        return jsonify({
            "symbol": symbol,
            "data": df.to_dict("records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations")
def api_recommendations():
    """Return ML-ranked buy and hold recommendations."""
    try:
        return jsonify(ml.get_recommendations())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/accuracy")
def api_accuracy():
    """Return prediction accuracy stats for all models."""
    try:
        return jsonify(ml.get_accuracy_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def api_predictions():
    """Return latest predictions for all symbols."""
    try:
        summary = ml.get_prediction_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/<symbol>")
def api_predictions_symbol(symbol):
    """Return latest predictions for one symbol."""
    symbol = symbol.upper()
    if symbol not in CRYPTO_ASSETS:
        return jsonify({"error": "Unknown symbol"}), 404
    try:
        df = db.get_latest_predictions(symbol)
        df["predicted_for"] = df["predicted_for"].astype(str)
        df["generated_at"]  = df["generated_at"].astype(str)
        return jsonify(df.to_dict("records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature_importance/<symbol>/<target>/<int:horizon>")
def api_feature_importance(symbol, target, horizon):
    """Return feature importances for a specific model."""
    symbol = symbol.upper()
    try:
        fi = ml.get_feature_importance(symbol, target, horizon)
        return jsonify(fi)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Trigger model retraining (async)."""
    def _retrain():
        try:
            ml.train_all()
            ml.predict_all()
        except Exception as e:
            print(f"[API] Retrain error: {e}")

    t = threading.Thread(target=_retrain, daemon=True)
    t.start()
    return jsonify({"status": "retraining started"})


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Trigger immediate data refresh."""
    def _do_refresh():
        with _refresh_lock:
            try:
                dc.refresh_all()
                ml.predict_all()
                _last_refresh["time"] = datetime.now(timezone.utc).isoformat()
                _last_refresh["status"] = "ok"
            except Exception as e:
                _last_refresh["status"] = f"error: {e}"

    t = threading.Thread(target=_do_refresh, daemon=True)
    t.start()
    return jsonify({"status": "refresh started"})


@app.route("/api/status")
def api_status():
    """Return system status."""
    counts = {}
    try:
        for sym in CRYPTO_ASSETS:
            counts[sym] = db.get_row_count(sym)
    except Exception as e:
        counts["error"] = str(e)

    import os
    from config import MODEL_DIR
    model_files = []
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    except Exception:
        pass

    return jsonify({
        "database_rows": counts,
        "models_trained": len(model_files),
        "last_refresh": _last_refresh,
        "assets": list(CRYPTO_ASSETS.keys()),
        "horizons": PREDICTION_HORIZONS,
    })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup():
    """Initialize DB, collect data, train models."""
    print("[STARTUP] Initializing database...")
    db.setup_database()

    print("[STARTUP] Checking data availability...")
    needs_bootstrap = False
    for sym in CRYPTO_ASSETS:
        n = db.get_row_count(sym)
        print(f"  {sym}: {n} rows")
        if n < 30:
            needs_bootstrap = True

    if needs_bootstrap:
        print("[STARTUP] Bootstrapping historical data (this may take a minute)...")
        dc.bootstrap_all(days=365)
    else:
        print("[STARTUP] Refreshing latest data...")
        dc.refresh_all()

    print("[STARTUP] Training ML models...")
    ml.train_all()

    print("[STARTUP] Generating initial predictions...")
    ml.predict_all()

    threading.Thread(target=_data_refresh_loop, daemon=True).start()
    threading.Thread(target=_prediction_refresh_loop, daemon=True).start()
    print(f"[STARTUP] Data refresh every {DATA_REFRESH_MINUTES}min, predictions every {PREDICTION_REFRESH_HOURS}hr.")


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
