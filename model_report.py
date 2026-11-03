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
    for t in ("funding_daily", "futures_snapshot", "options_snapshot"):
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        f.write(f"{t}: {n} rows\n")
    for r in c.execute("SELECT symbol, MIN([date]), COUNT(*) FROM funding_daily GROUP BY symbol").fetchall():
        f.write(f"  funding {r[0]} from {r[1]} ({r[2]} days)\n")
    for r in c.execute("SELECT TOP 9 symbol, funding*24*365*100, oi_usd/1e6, max_leverage FROM futures_snapshot ORDER BY ts DESC").fetchall():
        f.write(f"  perp {r[0]} funding {r[1]:.1f}% APR, OI ${r[2]:,.0f}M, max {r[3]}x\n")
    for r in c.execute("SELECT TOP 2 currency, put_call_ratio, dvol, oi_usd/1e9 FROM options_snapshot ORDER BY ts DESC").fetchall():
        f.write(f"  options {r[0]} P/C {r[1]:.2f} DVOL {r[2]} OI ${r[3]:.1f}B\n")
    p = c.execute("SELECT COUNT(*), MAX(generated_at) FROM predictions WHERE generated_at > DATEADD(minute,-30,GETUTCDATE())").fetchone()
    f.write(f"predictions last 30 min: {p[0]} (latest {p[1]})\n")
