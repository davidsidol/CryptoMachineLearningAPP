"""Clear outcomes scored since 2026-11-03 and re-score them with the corrected logic
(target = close of the predicted day; direction measured from the price at forecast time)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP); os.chdir(APP)
import database as db, ml_predictor as ml
removed = db.delete_outcomes_since("2026-11-03")
print(f"Removed {removed} outcomes scored with the old logic")
n = ml.evaluate_past_predictions()
print(f"Re-scored {n} outcomes")
with db.get_connection() as conn:
    for r in conn.cursor().execute("""
        SELECT target, horizon_days, COUNT(*), AVG(CAST(direction_correct AS FLOAT))*100
        FROM prediction_outcomes WHERE predicted_for >= '2026-11-03'
        GROUP BY target, horizon_days ORDER BY target, horizon_days""").fetchall():
        print(f"  {r[0]:<11} {r[1]}d  n={r[2]:<4} direction {r[3]:.1f}%")
