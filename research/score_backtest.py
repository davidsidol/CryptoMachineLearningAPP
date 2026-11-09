import pandas as pd, numpy as np
from scipy.stats import spearmanr
R=pd.read_csv("bt_preds.csv",parse_dates=["date"])
rows=[]
for (cfg,h),x in R.groupby(["cfg","h"],sort=False):
    x=x[x.act!=0]
    ok=(np.sign(x.pred)==np.sign(x.act))
    maj=(np.where(x.train_up>=.5,1,-1)==np.sign(x.act)); trd=(np.sign(x.trend)==np.sign(x.act))
    # day-level accuracy for an honest standard error (coins move together; windows overlap)
    da=ok.groupby(x.date).mean(); n_eff=len(da)/h; se=da.std()/np.sqrt(n_eff)*100
    skill=(1-(x.pred-x.act).abs().mean()/x.act.abs().mean())*100
    # ranking quality: per-day Spearman between prediction and actual ABSOLUTE return across coins, non-overlapping days
    days=sorted(x.date.unique())[::h]; ics=[]; sp=[]
    for d in days:
        g=x[x.date==d]
        if len(g)<6: continue
        ics.append(spearmanr(g.pred,g.act_abs).correlation)
        g=g.sort_values("pred"); sp.append(g.act_abs.tail(3).mean()-g.act_abs.head(3).mean())
    ics=np.array(ics); sp=np.array(sp)
    rows.append(dict(cfg=cfg,h=h,n=len(x),dir=ok.mean()*100,se=se,majority=maj.mean()*100,trend=trd.mean()*100,skill=skill,
        IC=np.nanmean(ics),IC_t=np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))),spread_pct=sp.mean()*100,spread_t=sp.mean()/(sp.std()/np.sqrt(len(sp))),periods=len(sp)))
S=pd.DataFrame(rows); pd.set_option("display.width",250)
print(S.round(2).to_string(index=False))
S.to_csv("bt_summary.csv",index=False)
