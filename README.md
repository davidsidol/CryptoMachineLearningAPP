# Crypto ML Forecasting Dashboard

A Python machine learning application that tracks and forecasts **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Solana (SOL)** price, volume, and market capitalisation using historical data stored in Microsoft SQL Server.

---

## Features

- **Live Data Ingestion** — pulls real-time and historical data from CoinMarketCap, Coinbase Exchange, and CoinGecko APIs
- **MSSQL Storage** — all market data and predictions persisted in SQL Server on a local `.\IDOLML` instance
- **Machine Learning** — 27 Gradient Boosting models (3 coins × 3 metrics × 3 horizons) trained with time-series cross-validation
- **Forecasts** — 1-day, 7-day, and 30-day predictions for price, volume, and market cap
- **Interactive Dashboard** — dark-themed Flask web app with Chart.js charts, forecast cards, and a comparison table
- **Auto-refresh** — background scheduler refreshes data and regenerates predictions every 60 minutes

---

## Dashboard Preview

### Per-coin view
- Live price card with 24h % change
- 1-day / 7-day / 30-day price forecast cards with confidence range and % vs current
- Volume 24h and market cap forecasts
- Price history chart with forecast overlay
- Volume bar chart
- Market cap line chart

### Compare view
- Normalised price comparison across all 3 chains (base = 100)
- Full forecast summary table with % change columns per horizon

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Flask 3 |
| ML Engine | scikit-learn (Gradient Boosting) |
| Database | Microsoft SQL Server 2019 (pyodbc) |
| Data Sources | CoinMarketCap API, Coinbase Exchange API, CoinGecko API |
| Frontend | Bootstrap 5, Chart.js 4, Bootstrap Icons |
| Scheduling | APScheduler |
| Model Persistence | joblib |

---

## Project Structure

```
crypto_ml/
├── app.py                  # Flask app and API routes
├── config.py               # API keys, DB connection, asset definitions
├── database.py             # MSSQL connection, schema, CRUD operations
├── data_collector.py       # CoinMarketCap, Coinbase, CoinGecko ingestion
├── ml_predictor.py         # Feature engineering, model training, prediction
├── start_app.py            # Main entry point (bootstrap + serve)
├── setup_db.py             # One-time DB + table creation
├── launch.bat              # Windows launch shortcut
├── requirements.txt        # Python dependencies
├── models/                 # Trained .pkl model files (27 total)
└── templates/
    └── index.html          # Single-page dashboard
```

---

## Database Schema (MSSQL — CryptoML)

| Table | Purpose |
|---|---|
| `crypto_prices` | Daily price, volume, market cap, % changes per symbol |
| `crypto_ohlcv` | OHLCV candles from Coinbase Exchange |
| `ml_models` | Model registry: MAE, RMSE, R², training row count |
| `predictions` | Stored forecasts with lower/upper confidence bounds |
| `collection_log` | Audit log of every API data pull |

---

## ML Model Details

- **Algorithm**: Gradient Boosting Regressor (300 estimators, depth 4, LR 0.05)
- **Features**: 8 lag periods (1/2/3/5/7/14/21/30 days), rolling mean/std (3/7/14/30 day windows), rate-of-change (7d/14d), calendar features (day-of-week, month)
- **Validation**: 3-fold time-series cross-validation (no data leakage)
- **Targets**: `price_usd`, `volume_24h`, `market_cap`
- **Horizons**: 1 day, 7 days, 30 days
- **Performance**: R² consistently 0.9996–0.9998 across all models

---

## Prerequisites

- Python 3.10+
- Microsoft SQL Server (local instance named `.\IDOLML`) — Windows Auth
- ODBC Driver 17 for SQL Server
- CoinMarketCap API key (free tier for live quotes)
- Internet access (Coinbase and CoinGecko are free/public)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/davidsidol/CryptoMachineLearningAPP.git
cd CryptoMachineLearningAPP

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure API keys
#    Edit config.py and set your CoinMarketCap API key
#    DB_SERVER defaults to .\IDOLML — update if your instance differs

# 4. Run the application
python start_app.py
```

On first run, `start_app.py` will:
1. Create the `CryptoML` database and all tables
2. Bootstrap 365 days of historical data (CoinGecko + Coinbase)
3. Fetch the current live snapshot (CoinMarketCap)
4. Train all 27 ML models
5. Generate initial predictions
6. Start the web server

---

## Usage

Open your browser to **http://localhost:5001**

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main dashboard |
| GET | `/api/status` | System status (DB rows, models trained) |
| GET | `/api/predictions` | Latest predictions for all symbols |
| GET | `/api/predictions/<SYMBOL>` | Predictions for one symbol (BTC/ETH/SOL) |
| GET | `/api/prices/<SYMBOL>?days=90` | Historical price data |
| GET | `/api/ohlcv/<SYMBOL>?days=90` | Historical OHLCV candle data |
| GET | `/api/feature_importance/<SYM>/<TARGET>/<HORIZON>` | Model feature importances |
| POST | `/api/refresh` | Trigger immediate data refresh |
| POST | `/api/retrain` | Trigger full model retraining |

---

## Configuration (`config.py`)

```python
DB_SERVER            = r".\IDOLML"          # SQL Server instance name
DB_NAME              = "CryptoML"            # Database name
COINMARKETCAP_API_KEY = "your_key_here"     # CMC API key
PREDICTION_HORIZONS  = [1, 7, 30]           # Forecast horizons (days)
DATA_REFRESH_MINUTES = 60                    # Background refresh interval
FLASK_PORT           = 5001                  # Web server port
```

---

## Data Sources

| Source | Data | Auth Required |
|---|---|---|
| [CoinMarketCap](https://coinmarketcap.com/api/) | Live price, volume, market cap, % changes | API key (free tier) |
| [Coinbase Exchange](https://docs.cdp.coinbase.com/) | OHLCV daily candles (up to 300 days) | None (public) |
| [CoinGecko](https://www.coingecko.com/en/api) | 365-day historical price, volume, market cap | None (free tier) |

---

## License

MIT License — free to use, modify, and distribute.
