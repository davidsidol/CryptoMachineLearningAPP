import requests, time, pandas as pd, datetime as dt
H={"User-Agent":"cryptoml-research"}
def coinbase(prod, start="2017-01-01"):
    out=[]; s=pd.Timestamp(start,tz="UTC"); end=pd.Timestamp.now(tz="UTC").normalize()
    while s<end:
        e=min(s+pd.Timedelta(days=299),end)
        for a in range(4):
            r=requests.get(f"https://api.exchange.coinbase.com/products/{prod}/candles",params={"granularity":86400,"start":s.isoformat(),"end":e.isoformat()},headers=H,timeout=30)
            if r.status_code==200: break
            time.sleep(2+a*3)
        if r.status_code==200: out+=r.json()
        s=e+pd.Timedelta(days=1); time.sleep(0.25)
    d=pd.DataFrame(out,columns=["t","low","high","open","close","vol"])
    d["date"]=pd.to_datetime(d.t,unit="s"); d["volume_usd"]=d.vol*d.close
    return d[["date","open","high","low","close","volume_usd"]].drop_duplicates("date").sort_values("date")
def hl(coin):
    r=requests.post("https://api.hyperliquid.xyz/info",json={"type":"candleSnapshot","req":{"coin":coin,"interval":"1d","startTime":0,"endTime":int(time.time()*1000)}},timeout=30).json()
    d=pd.DataFrame(r); d["date"]=pd.to_datetime(d.t,unit="ms")
    for c in "ohlcv": d[c]=d[c].astype(float)
    d["volume_usd"]=d.v*d.c
    return d.rename(columns={"o":"open","h":"high","l":"low","c":"close"})[["date","open","high","low","close","volume_usd"]].sort_values("date")
src={"BTC":("cb","BTC-USD"),"ETH":("cb","ETH-USD"),"SOL":("cb","SOL-USD"),"AVAX":("cb","AVAX-USD"),"NEAR":("cb","NEAR-USD"),"LINK":("cb","LINK-USD"),"TAO":("hl","TAO"),"HYPE":("hl","HYPE"),"XMR":("hl","XMR")}
frames=[]
for sym,(k,p) in src.items():
    d=coinbase(p) if k=="cb" else hl(p); d["symbol"]=sym; d["source"]=k; frames.append(d)
    print(sym,k,len(d),d.date.min().date(),d.date.max().date(),flush=True)
a=pd.concat(frames); today=pd.Timestamp.utcnow().normalize().tz_localize(None)
a=a[a.date<today]   # drop today's unfinished candle
a.to_csv("daily.csv",index=False)
# funding: extend BTC etc back as far as available
def funding(coin):
    rows=[]; s=0
    while True:
        pg=requests.post("https://api.hyperliquid.xyz/info",json={"type":"fundingHistory","coin":coin,"startTime":s},timeout=30).json()
        if not pg: break
        rows+=pg
        if len(pg)<500: break
        s=int(pg[-1]["time"])+1; time.sleep(0.15)
    d=pd.DataFrame(rows); d["date"]=pd.to_datetime(d.time,unit="ms").dt.normalize(); d["f"]=d.fundingRate.astype(float); d["p"]=d.premium.astype(float)
    g=d.groupby("date").agg(funding_avg=("f","mean"),premium_avg=("p","mean")).reset_index(); g["symbol"]=coin; return g
ff=[]
for sym in src:
    try: g=funding(sym); ff.append(g); print("funding",sym,len(g),g.date.min().date(),flush=True)
    except Exception as e: print("funding fail",sym,e)
pd.concat(ff).to_csv("funding.csv",index=False)
