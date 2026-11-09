"""
Startup script — Flask binds immediately, all heavy work runs in background.
The dashboard shows a loading banner until initialisation is complete.
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys, threading, time
import database as db
import ml_predictor as ml
import data_collector as dc
from config import CRYPTO_ASSETS, FLASK_PORT, DATA_REFRESH_MINUTES, PREDICTION_REFRESH_HOURS

# ── Shared startup state (read by Flask routes) ──────────────────────────────
startup_state = {
    "done":    False,
    "stage":   "Starting up…",
    "error":   None,
}

def _run_startup():
    """All heavy initialisation — runs in a daemon thread so Flask starts first."""
    try:
        # 1. Database
        startup_state["stage"] = "Setting up database…"
        db.setup_database()

        # 2. Data
        new_syms = [s for s in CRYPTO_ASSETS if db.get_row_count(s) < 30]
        if new_syms:
            startup_state["stage"] = f"Bootstrapping 365 days of history for {', '.join(new_syms)}…"
            dc.bootstrap_symbols(new_syms, days=365)
        startup_state["stage"] = "Loading Fear & Greed history…"
        dc.ensure_fear_greed_history()
        startup_state["stage"] = "Loading futures funding history…"
        dc.ensure_funding_history()
        startup_state["stage"] = "Loading full daily price history…"
        dc.ensure_daily_history()
        startup_state["stage"] = "Refreshing latest market data…"
        dc.refresh_all()

        # 3. Train any model that is missing or built by an older version
        expected = [(sym, tgt, h) for sym in CRYPTO_ASSETS
                    for tgt in ml.PER_COIN_TARGETS for h in ml.PREDICTION_HORIZONS]
        missing = [e for e in expected if not ml.model_is_current(*e)]
        if missing:
            startup_state["stage"] = f"Training {len(missing)} volume / market-cap models…"
            ml.train_all(only_missing=True)
        if not all(ml.pooled_is_current(h) for h in ml.PREDICTION_HORIZONS):
            startup_state["stage"] = "Training full-history price model…"
            ml.train_pooled_all()

        # 3b. Hidden up/down test models (trained once, never shown)
        try:
            ml.ensure_shadow_models()
        except Exception as e:
            print(f"[SHADOW] {e}")

        # 4. Predictions
        startup_state["stage"] = "Generating predictions…"
        ml.predict_all()
        try:
            ml.shadow_predict_all()
        except Exception as e:
            print(f"[SHADOW] {e}")

        # 5. Evaluate past predictions (score any that have now matured)
        startup_state["stage"] = "Evaluating past prediction accuracy…"
        ml.evaluate_past_predictions()

        startup_state["stage"] = "Ready"
        startup_state["done"]  = True

    except Exception as e:
        startup_state["stage"] = f"Startup error: {e}"
        startup_state["error"] = str(e)
        print(f"[STARTUP ERROR] {e}")
        return

    print(f"\n{'='*52}")
    print(f"  Crypto ML Dashboard is ready at:")
    print(f"  http://localhost:{FLASK_PORT}")
    print(f"{'='*52}\n")


def _data_refresh_loop():
    """Pull new market data every DATA_REFRESH_MINUTES, then evaluate past predictions."""
    while True:
        time.sleep(DATA_REFRESH_MINUTES * 60)
        if not startup_state["done"]:
            continue
        try:
            print("[REFRESH] Refreshing market data…")
            dc.refresh_all()
            ml.evaluate_past_predictions()
            ml.score_shadow()
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")


def _weekly_retrain_loop():
    """Retrain once the models are more than a week old (checked every 6 hours)."""
    while True:
        time.sleep(6 * 3600)
        if not startup_state["done"]:
            continue
        try:
            if ml.retrain_if_stale(7):
                ml.predict_all()
        except Exception as e:
            print(f"[RETRAIN ERROR] {e}")


def _prediction_refresh_loop():
    """Regenerate ML predictions every PREDICTION_REFRESH_HOURS."""
    while True:
        time.sleep(PREDICTION_REFRESH_HOURS * 3600)
        if not startup_state["done"]:
            continue
        try:
            print("[PREDICT] Regenerating predictions…")
            try:
                dc.refresh_daily_candles()
            except Exception as e:
                print(f"[PREDICT] Daily candle refresh error: {e}")
            ml.predict_all()
            ml.shadow_predict_all()
        except Exception as e:
            print(f"[PREDICT ERROR] {e}")


if __name__ == "__main__":
    # Patch the Flask app to expose startup_state
    from app import app, _last_refresh
    app.config["STARTUP_STATE"] = startup_state

    # Patch index route to show loading banner if not ready
    from flask import jsonify
    original_index = app.view_functions.get("index")

    @app.route("/api/startup_status")
    def api_startup_status():
        return jsonify(startup_state)

    # Fire heavy work in background
    t_init = threading.Thread(target=_run_startup, daemon=True)
    t_init.start()

    # Fire periodic refresh threads
    threading.Thread(target=_data_refresh_loop, daemon=True).start()
    threading.Thread(target=_prediction_refresh_loop, daemon=True).start()
    threading.Thread(target=_weekly_retrain_loop, daemon=True).start()

    print(f"[STARTUP] Flask binding on port {FLASK_PORT} — initialisation running in background…")
    print(f"[STARTUP] Open http://localhost:{FLASK_PORT} — a loading banner will show until ready.\n")

    try:
        from waitress import serve
        print(f"[STARTUP] Serving with Waitress on 0.0.0.0:{FLASK_PORT}")
        serve(app, host="0.0.0.0", port=FLASK_PORT, threads=8)
    except ImportError:
        print("[STARTUP] Waitress not installed — falling back to the Flask dev server")
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
