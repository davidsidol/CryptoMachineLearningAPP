"""
Configuration for Crypto ML Forecasting Application
"""

# Database Configuration
DB_SERVER = r".\IDOLML"
DB_NAME = "CryptoML"
DB_DRIVER = "ODBC Driver 17 for SQL Server"
DB_CONN_STR = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
DB_MASTER_CONN_STR = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    "DATABASE=master;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# API Keys
COINMARKETCAP_API_KEY = "aace0afa29b1424f92ae25c0e2ab9f45"
COINMARKETCAP_BASE_URL = "https://pro-api.coinmarketcap.com"

COINBASE_API_KEY = "1DhiiykwRtKCdveXxyRmnpuqXgYdR44g"
COINBASE_BASE_URL = "https://api.exchange.coinbase.com"

# Crypto Assets to Track
CRYPTO_ASSETS = {
    "BTC": {"name": "Bitcoin",  "cmc_id": 1,    "coinbase_id": "BTC-USD", "color": "#F7931A"},
    "ETH": {"name": "Ethereum", "cmc_id": 1027,  "coinbase_id": "ETH-USD", "color": "#627EEA"},
    "SOL": {"name": "Solana",   "cmc_id": 5426,  "coinbase_id": "SOL-USD", "color": "#9945FF"},
}

# ML Configuration
PREDICTION_HORIZONS = [1, 7, 30]   # days
MODEL_DIR = "models"
MIN_TRAINING_ROWS = 60              # minimum rows before training

# Flask Configuration
FLASK_SECRET_KEY = "crypto_ml_secret_key_2024"
FLASK_PORT = 5001
FLASK_DEBUG = False

# Scheduler - how often to refresh live data (minutes)
DATA_REFRESH_MINUTES = 60
