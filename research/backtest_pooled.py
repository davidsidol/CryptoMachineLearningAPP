import pandas as pd, numpy as np, warnings, sys, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
D=pd.read_csv("daily.csv",parse_dates=["date"])
FNG=pd.read_csv("/mnt/user-data/uploads/CryptoML/logs/export/hist_fng.csv",parse_dates=["date"]).set_index("date").sort_index()["value"].astype(float)
FU=pd.read_csv("funding.csv",parse_dates=["date"])
SYMS=list(D.symbol.unique()); HS=(1,3,7)
px=D.pivot(index="date",columns="symbol",values="close").asfreq("D").ffill(limit=3)
vol=D.pivot(index="date",columns="symbol",values="volume_usd").asfreq("D")
hi=D.pivot(index="date",columns="symbol",values="high").asfreq("D"); lo=D.pivot(index="date",columns="symbol",values="low").asfreq("D")
def feats(s):
    p=px[s]; r1=p.pct_change(fill_method=None); f=pd.DataFrame(index=px.index)
    for k in (1,2,3,5,7,14,21,30): f[f"ret_{k}d"]=p.pct_change(k,fill_method=None)
    for w in (7,14,30): f[f"vol_{w}d"]=r1.rolling(w).std(); f[f"px_vs_ma{w}"]=p/p.rolling(w).mean()-1
    f["px_vs_high30"]=p/p.rolling(30).max()-1; f["px_vs_low30"]=p/p.rolling(30).min()-1
    f["px_vs_high90"]=p/p.rolling(90).max()-1; f["vol_ratio"]=f.vol_7d/f.vol_30d
    f["range_7d"]=((hi[s]-lo[s])/p).rolling(7).mean()
    v=vol[s].replace(0,np.nan)
    f["volu_chg_1d"]=v.pct_change(1,fill_method=None).clip(-5,5); f["volu_vs_ma7"]=(v/v.rolling(7).mean()-1).clip(-5,5); f["volu_vs_ma30"]=(v/v.rolling(30).mean()-1).clip(-5,5)
    f["dow"]=f.index.dayofweek
    # context
    b=px["BTC"]; br=b.pct_change(fill_method=None)
    for k in (1,3,7,14,30): f[f"btc_ret_{k}d"]=b.pct_change(k,fill_method=None)
    f["btc_vol_14d"]=br.rolling(14).std(); f["btc_px_vs_ma30"]=b/b.rolling(30).mean()-1
    f["rel_ret_7d"]=f.ret_7d-f.btc_ret_7d; f["rel_ret_30d"]=f.ret_30d-f.btc_ret_30d
    f["beta_60"]=r1.rolling(60).cov(br)/br.rolling(60).var()
    g=FNG.reindex(f.index).ffill(limit=3); f["fng"]=g; f["fng_chg_7d"]=g-g.shift(7); f["fng_vs_ma30"]=g-g.rolling(30).mean()
    fu=FU[FU.symbol==s].set_index("date")["funding_avg"].reindex(f.index)*24*365
    f["fund_1d"]=fu; f["fund_7d"]=fu.rolling(7,min_periods=1).mean(); f["fund_z30"]=(fu-fu.rolling(30,min_periods=10).mean())/fu.rolling(30,min_periods=10).std()
    f["symbol"]=s; f["vol30"]=f.vol_30d
    for h in HS:
        f[f"y{h}"]=p.shift(-h)/p-1
        f[f"yb{h}"]=b.shift(-h)/b-1
    return f.reset_index()
A=pd.concat([feats(s) for s in SYMS]).replace([np.inf,-np.inf],np.nan)
A=A.dropna(subset=["ret_30d","vol_30d"])
# cross-sectional ranks (among coins with data that day)
for c in ("ret_7d","ret_30d","vol_30d","volu_vs_ma30","rel_ret_7d"):
    A["xs_"+c]=A.groupby("date")[c].rank(pct=True)
META=["date","symbol","vol30"]+[f"y{h}" for h in HS]+[f"yb{h}" for h in HS]
FE=[c for c in A.columns if c not in META]
SHORT_START=pd.Timestamp("2025-03-05")
last=A.date.max(); test_start=last-pd.Timedelta(days=420)
def model(): return HistGradientBoostingRegressor(loss="absolute_error",max_depth=3,learning_rate=0.03,max_iter=200,min_samples_leaf=40,l2_regularization=1.0,random_state=42)
def run(name,h,pooled,long,relative,scaled):
    out=[]
    fs=test_start
    while fs<last:
        fe=fs+pd.Timedelta(days=30); cut=fs-pd.Timedelta(days=h)
        tr=A[(A.date<cut)&A[f"y{h}"].notna()]; te=A[(A.date>=fs)&(A.date<fe)&A[f"y{h}"].notna()]
        if not long: tr=tr[tr.date>=SHORT_START]
        def tgt(d):
            y=d[f"y{h}"]-(d[f"yb{h}"] if relative else 0)
            return y/(d.vol30*np.sqrt(h)) if scaled else y
        groups=[None] if pooled else SYMS
        for g in groups:
            a=tr if g is None else tr[tr.symbol==g]; b=te if g is None else te[te.symbol==g]
            if relative: a=a[a.symbol!="BTC"]; b=b[b.symbol!="BTC"]
            if len(a)<150 or b.empty: continue
            cols=[c for c in FE if a[c].notna().sum()>=20]
            m=model().fit(a[cols].values,tgt(a).values)
            p=m.predict(b[cols].values)
            if scaled: p=p*(b.vol30.values*np.sqrt(h))
            o=b[["date","symbol"]].copy(); o["pred"]=p; o["act"]=(b[f"y{h}"]-(b[f"yb{h}"] if relative else 0)).values
            o["act_abs"]=b[f"y{h}"].values; o["trend"]=(b[f"ret_{h}d"]-(b[f"btc_ret_{h}d"] if relative else 0)).values
            o["train_up"]=(tgt(a)>0).mean()
            out.append(o)
        fs=fe
    o=pd.concat(out); o["cfg"]=name; o["h"]=h; return o
CFG={ # name: pooled,long,relative,scaled
 "B per-coin, 20mo, clean":(False,False,False,False),
 "C per-coin, full history":(False,True,False,False),
 "D pooled, full history":(True,True,False,True),
 "E pooled, full, vs BTC":(True,True,True,True),
 "E0 per-coin, 20mo, vs BTC":(False,False,True,False),
}
res=[]
for n,(a,b,c,d) in CFG.items():
    for h in HS:
        t=time.time(); res.append(run(n,h,a,b,c,d)); print(n,h,round(time.time()-t),"s",flush=True)
R=pd.concat(res); R.to_csv("bt_preds.csv",index=False)
