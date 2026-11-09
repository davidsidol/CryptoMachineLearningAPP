"""Run ml_predictor backtests offline against CSV exports from the server."""
import sys, types, contextlib, pandas as pd
D = sys.argv[1]
sys.path.insert(0, '/home/claude/repo')
P = pd.read_csv(f'{D}/hist_prices.csv', parse_dates=['timestamp'])
F = pd.read_csv(f'{D}/hist_fng.csv')
U = pd.read_csv(f'{D}/hist_funding.csv')
NOW = P.timestamp.max()
def gph(sym, days=365):
    x = P[(P.symbol == sym) & (P.timestamp >= NOW - pd.Timedelta(days=days))]
    return x.drop(columns='symbol').sort_values('timestamp').reset_index(drop=True)
def gfg(days=31):
    return F.sort_values('date', ascending=False).head(days).reset_index(drop=True)
def gfd(sym, days=730):
    x = U[U.symbol == sym].copy(); x['date'] = pd.to_datetime(x['date'])
    x = x[x.date >= NOW.normalize() - pd.Timedelta(days=days)]
    return x[['date', 'funding_avg', 'premium_avg']].sort_values('date').reset_index(drop=True)
fake = types.ModuleType('database')
fake.get_price_history = gph; fake.get_fear_greed = gfg; fake.get_funding_daily = gfd
fake.get_connection = contextlib.contextmanager(lambda master=False: (yield None))
sys.modules['database'] = fake
import warnings; warnings.filterwarnings('ignore')
import ml_predictor as ml
