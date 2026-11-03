"""
Measure which market-context inputs improve the price models.
Writes logs/ablation.txt — run after the funding history backfill has finished.
"""
import warnings
warnings.filterwarnings("ignore")
import ml_predictor as ml

df = ml.ablation_report("price_usd")
df.to_csv(r"logs\ablation.csv", index=False)
with open(r"logs\ablation.txt", "w") as f:
    f.write("Average across coins (out-of-sample, price models)\n")
    f.write(df.groupby(["inputs", "horizon"])[["skill", "directional"]].mean().round(2)
              .unstack("horizon").to_string() + "\n\n")
    f.write("Skill vs no-change by coin (%), 7-day horizon\n")
    f.write(df[df.horizon == 7].pivot(index="symbol", columns="inputs", values="skill").round(1).to_string() + "\n\n")
    f.write("Directional accuracy by coin (%), 7-day horizon\n")
    f.write(df[df.horizon == 7].pivot(index="symbol", columns="inputs", values="directional").round(1).to_string() + "\n")
print("done")
