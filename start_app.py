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
from config import CRYPTO_ASSETS, FLASK_PORT, DATA_REFRESH_MINUTES

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
        needs_bootstrap = any(db.get_row_count(s) < 30 for s in CRYPTO_ASSETS)
        if needs_bootstrap:
            startup_state["stage"] = "Bootstrapping 365 days of historical data…"
            dc.bootstrap_all(days=365)
        else:
            startup_state["stage"] = "Refreshing latest market data…"
            dc.refresh_all()

        # 3. Check whether saved models exist — skip training if all 27 present
        model_dir = ml.MODEL_DIR
        expected = [
            f"{sym}_{tgt}_{h}d.pkl"
            for sym in CRYPTO_ASSETS
            for tgt in ml.TARGETS
            for h in ml.PREDICTION_HORIZONS
        ]
        missing = [f for f in expected if not os.path.exists(os.path.join(model_dir, f))]

        if missing:
            startup_state["stage"] = f"Training ML models (0 / {len(expected)})…"
            ml.train_all()
        else:
            startup_state["stage"] = "ML models already trained — skipping retrain…"
            time.sleep(1)

        # 4. Predictions
        startup_state["stage"] = "Generating predictions…"
        ml.predict_all()

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
        print(f"[STARTUP ERROR] {e}")


def _background_refresh():
    """Periodic data + prediction refresh (every DATA_REFRESH_MINUTES)."""
    while True:
        time.sleep(DATA_REFRESH_MINUTES * 60)
        if not startup_state["done"]:
            continue
        try:
            print("[REFRESH] Refreshing data and predictions…")
            dc.refresh_all()
            ml.predict_all()
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")


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

    # Fire periodic refresh in background
    t_refresh = threading.Thread(target=_background_refresh, daemon=True)
    t_refresh.start()

    print(f"[STARTUP] Flask binding on port {FLASK_PORT} — initialisation running in background…")
    print(f"[STARTUP] Open http://localhost:{FLASK_PORT} — a loading banner will show until ready.\n")

    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
