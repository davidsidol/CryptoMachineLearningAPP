import pyodbc, config
c = pyodbc.connect(config.DB_CONN_STR)
rows = c.execute("""
  SELECT symbol, target, horizon_days, mae*100, baseline_mae*100, skill_score*100,
         directional_acc, up_share, training_rows
  FROM ml_models WHERE is_active=1 AND model_version=3
  ORDER BY target, horizon_days, symbol""").fetchall()
with open(r"logs\model_report.txt", "w") as f:
    f.write(f"{len(rows)} active v3 models\n")
    f.write("sym,target,h,mae%,baseline%,skill%,dir%,up%,rows\n")
    for r in rows:
        f.write(",".join([r[0], r[1], str(r[2])] + [f"{x:.2f}" for x in r[3:8]] + [str(r[8])]) + "\n")
    fg = c.execute("SELECT COUNT(*), MIN([date]), MAX([date]) FROM fear_greed").fetchone()
    f.write(f"fear_greed rows={fg[0]} from {fg[1]} to {fg[2]}\n")
    last = c.execute("SELECT TOP 3 [date], value, classification FROM fear_greed ORDER BY [date] DESC").fetchall()
    f.write("latest: " + "; ".join(f"{d} {v} {l}" for d, v, l in last) + "\n")
    p = c.execute("SELECT COUNT(*), MAX(generated_at) FROM predictions WHERE generated_at > DATEADD(minute,-30,GETUTCDATE())").fetchone()
    f.write(f"predictions last 30 min: {p[0]} (latest {p[1]})\n")
