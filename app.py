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
    DATA_REFRESH_MINUTES
)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Background refresh state
_refresh_lock = threading.Lock()
_last_refresh = {"time": None, "status": "never"}


def _background_refresh():
    """Periodically refresh data and regenerate predictions."""
    while True:
        time.sleep(DATA_REFRESH_MINUTES * 60)
        with _refresh_lock:
            try:
                print("[APP] Background refresh starting...")
                dc.refresh_all()
                ml.predict_all()
                _last_refresh["time"] = datetime.now(timezone.utc).isoformat()
                _last_refresh["status"] = "ok"
                print("[APP] Background refresh complete.")
            except Exception as e:
                _last_refresh["status"] = f"error: {e}"
                print(f"[APP] Background refresh error: {e}")


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

    return render_template(
        "index.html",
        assets=CRYPTO_ASSETS,
        prices=prices,
        predictions=predictions,
        horizons=PREDICTION_HORIZONS,
        last_refresh=_last_refresh,
    )


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

    # Start background refresh thread
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    print(f"[STARTUP] Background refresh every {DATA_REFRESH_MINUTES} minutes.")


if __name__ == "__main__":
    startup()
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
