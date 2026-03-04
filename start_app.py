"""
Startup script - run this to launch the Crypto ML application.
Skips re-bootstrapping if data already exists.
"""
import warnings
warnings.filterwarnings("ignore")

import database as db
import ml_predictor as ml
import data_collector as dc
from config import CRYPTO_ASSETS, FLASK_PORT
import threading
import time

def startup_skip_bootstrap():
    """Skip bootstrap if data is already loaded, just refresh + retrain."""
    db.setup_database()

    needs_train = False
    for sym in CRYPTO_ASSETS:
        n = db.get_row_count(sym)
        print(f"  {sym}: {n} rows in DB")
        if n < 30:
            needs_train = True

    if needs_train:
        print("[STARTUP] Bootstrapping data...")
        dc.bootstrap_all(days=365)
    else:
        print("[STARTUP] Refreshing latest data...")
        dc.refresh_all()

    print("[STARTUP] Training ML models...")
    ml.train_all()

    print("[STARTUP] Generating predictions...")
    ml.predict_all()

if __name__ == "__main__":
    from app import app, _background_refresh, DATA_REFRESH_MINUTES

    startup_skip_bootstrap()

    # Start background refresh thread
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    print(f"[STARTUP] Background refresh every {DATA_REFRESH_MINUTES} min")

    print(f"\n{'='*50}")
    print(f"  Crypto ML Dashboard running at:")
    print(f"  http://localhost:{FLASK_PORT}")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
