"""
Database connection and operations for Crypto ML
"""
import pyodbc
import pandas as pd
from datetime import datetime
from contextlib import contextmanager
from config import DB_CONN_STR, DB_MASTER_CONN_STR, DB_NAME


@contextmanager
def get_connection(master=False):
    """Context manager for database connections."""
    conn_str = DB_MASTER_CONN_STR if master else DB_CONN_STR
    conn = pyodbc.connect(conn_str, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def setup_database():
    """Create the CryptoML database and all required tables."""
    print(f"[DB] Setting up database '{DB_NAME}'...")

    # Create database if it doesn't exist
    with get_connection(master=True) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = '{DB_NAME}')
            BEGIN
                CREATE DATABASE [{DB_NAME}]
            END
        """)
        print(f"[DB] Database '{DB_NAME}' ready.")

    # Create tables
    with get_connection() as conn:
        cursor = conn.cursor()

        # Historical price/volume/marketcap data
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='crypto_prices' AND xtype='U')
            CREATE TABLE crypto_prices (
                id              BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol          NVARCHAR(10)   NOT NULL,
                timestamp       DATETIME2      NOT NULL,
                price_usd       FLOAT          NOT NULL,
                volume_24h      FLOAT,
                market_cap      FLOAT,
                pct_change_1h   FLOAT,
                pct_change_24h  FLOAT,
                pct_change_7d   FLOAT,
                circulating_supply FLOAT,
                source          NVARCHAR(50)   DEFAULT 'coinmarketcap',
                created_at      DATETIME2      DEFAULT GETUTCDATE(),
                CONSTRAINT UQ_symbol_timestamp UNIQUE (symbol, timestamp)
            )
        """)

        # Store individual OHLCV candles (from Coinbase)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='crypto_ohlcv' AND xtype='U')
            CREATE TABLE crypto_ohlcv (
                id          BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol      NVARCHAR(10)  NOT NULL,
                timestamp   DATETIME2     NOT NULL,
                open_price  FLOAT,
                high_price  FLOAT,
                low_price   FLOAT,
                close_price FLOAT,
                volume      FLOAT,
                granularity INT           DEFAULT 86400,
                source      NVARCHAR(50)  DEFAULT 'coinbase',
                created_at  DATETIME2     DEFAULT GETUTCDATE(),
                CONSTRAINT UQ_ohlcv_sym_ts UNIQUE (symbol, timestamp, granularity)
            )
        """)

        # ML model registry
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ml_models' AND xtype='U')
            CREATE TABLE ml_models (
                id              INT IDENTITY(1,1) PRIMARY KEY,
                symbol          NVARCHAR(10)  NOT NULL,
                target          NVARCHAR(50)  NOT NULL,
                horizon_days    INT           NOT NULL,
                model_file      NVARCHAR(500),
                mae             FLOAT,
                rmse            FLOAT,
                r2_score        FLOAT,
                training_rows   INT,
                trained_at      DATETIME2     DEFAULT GETUTCDATE(),
                is_active       BIT           DEFAULT 1
            )
        """)

        # Prediction storage
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='predictions' AND xtype='U')
            CREATE TABLE predictions (
                id              BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol          NVARCHAR(10)  NOT NULL,
                target          NVARCHAR(50)  NOT NULL,
                horizon_days    INT           NOT NULL,
                predicted_for   DATETIME2     NOT NULL,
                predicted_value FLOAT         NOT NULL,
                lower_bound     FLOAT,
                upper_bound     FLOAT,
                model_id        INT,
                generated_at    DATETIME2     DEFAULT GETUTCDATE(),
                CONSTRAINT UQ_pred UNIQUE (symbol, target, horizon_days, predicted_for, generated_at)
            )
        """)

        # Data collection log
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='collection_log' AND xtype='U')
            CREATE TABLE collection_log (
                id          INT IDENTITY(1,1) PRIMARY KEY,
                run_at      DATETIME2     DEFAULT GETUTCDATE(),
                source      NVARCHAR(50),
                symbol      NVARCHAR(10),
                records_inserted INT,
                status      NVARCHAR(20),
                error_msg   NVARCHAR(MAX)
            )
        """)

        print("[DB] All tables created/verified.")


def upsert_price(symbol: str, timestamp: datetime, price: float,
                 volume: float = None, market_cap: float = None,
                 pct_1h: float = None, pct_24h: float = None,
                 pct_7d: float = None, circ_supply: float = None,
                 source: str = "coinmarketcap"):
    """Insert or update a price record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            MERGE crypto_prices AS target
            USING (SELECT ? AS symbol, ? AS timestamp) AS source
            ON target.symbol = source.symbol AND target.timestamp = source.timestamp
            WHEN MATCHED THEN
                UPDATE SET price_usd=?, volume_24h=?, market_cap=?,
                           pct_change_1h=?, pct_change_24h=?, pct_change_7d=?,
                           circulating_supply=?
            WHEN NOT MATCHED THEN
                INSERT (symbol, timestamp, price_usd, volume_24h, market_cap,
                        pct_change_1h, pct_change_24h, pct_change_7d,
                        circulating_supply, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, symbol, timestamp,
             price, volume, market_cap, pct_1h, pct_24h, pct_7d, circ_supply,
             symbol, timestamp, price, volume, market_cap, pct_1h, pct_24h, pct_7d,
             circ_supply, source)


def upsert_ohlcv(symbol: str, timestamp: datetime, open_p: float,
                 high_p: float, low_p: float, close_p: float,
                 volume: float, granularity: int = 86400, source: str = "coinbase"):
    """Insert or update OHLCV candle."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            MERGE crypto_ohlcv AS target
            USING (SELECT ? AS symbol, ? AS timestamp, ? AS granularity) AS source
            ON target.symbol = source.symbol
               AND target.timestamp = source.timestamp
               AND target.granularity = source.granularity
            WHEN MATCHED THEN
                UPDATE SET open_price=?, high_price=?, low_price=?, close_price=?, volume=?
            WHEN NOT MATCHED THEN
                INSERT (symbol, timestamp, open_price, high_price, low_price,
                        close_price, volume, granularity, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, symbol, timestamp, granularity,
             open_p, high_p, low_p, close_p, volume,
             symbol, timestamp, open_p, high_p, low_p, close_p, volume, granularity, source)


def get_price_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical price data for a symbol."""
    with get_connection() as conn:
        query = """
            SELECT timestamp, price_usd, volume_24h, market_cap,
                   pct_change_1h, pct_change_24h, pct_change_7d, circulating_supply
            FROM crypto_prices
            WHERE symbol = ?
              AND timestamp >= DATEADD(day, -?, GETUTCDATE())
            ORDER BY timestamp ASC
        """
        return pd.read_sql(query, conn, params=[symbol, days],
                           parse_dates=["timestamp"])


def get_ohlcv_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV history for a symbol."""
    with get_connection() as conn:
        query = """
            SELECT timestamp, open_price, high_price, low_price,
                   close_price, volume
            FROM crypto_ohlcv
            WHERE symbol = ? AND granularity = 86400
              AND timestamp >= DATEADD(day, -?, GETUTCDATE())
            ORDER BY timestamp ASC
        """
        return pd.read_sql(query, conn, params=[symbol, days],
                           parse_dates=["timestamp"])


def save_prediction(symbol: str, target: str, horizon: int,
                    predicted_for: datetime, value: float,
                    lower: float = None, upper: float = None,
                    model_id: int = None):
    """Save a prediction to the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
                (symbol, target, horizon_days, predicted_for,
                 predicted_value, lower_bound, upper_bound, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, symbol, target, horizon, predicted_for, value, lower, upper, model_id)


def get_latest_predictions(symbol: str = None) -> pd.DataFrame:
    """Get the most recent predictions for each symbol/horizon."""
    with get_connection() as conn:
        sym_filter = f"AND symbol = '{symbol}'" if symbol else ""
        query = f"""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY symbol, target, horizon_days
                    ORDER BY generated_at DESC
                ) AS rn
                FROM predictions
                WHERE 1=1 {sym_filter}
            )
            SELECT symbol, target, horizon_days, predicted_for,
                   predicted_value, lower_bound, upper_bound, generated_at
            FROM ranked
            WHERE rn = 1
            ORDER BY symbol, horizon_days
        """
        return pd.read_sql(query, conn, parse_dates=["predicted_for", "generated_at"])


def get_latest_prices() -> pd.DataFrame:
    """Get the most recent price for each symbol."""
    with get_connection() as conn:
        query = """
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY symbol ORDER BY timestamp DESC
                ) AS rn
            FROM crypto_prices
            )
            SELECT symbol, timestamp, price_usd, volume_24h, market_cap,
                   pct_change_24h, pct_change_7d
            FROM ranked
            WHERE rn = 1
        """
        return pd.read_sql(query, conn, parse_dates=["timestamp"])


def log_collection(source: str, symbol: str, records: int,
                   status: str, error: str = None):
    """Log a data collection run."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO collection_log (source, symbol, records_inserted, status, error_msg)
            VALUES (?, ?, ?, ?, ?)
        """, source, symbol, records, status, error)


def get_row_count(symbol: str) -> int:
    """Return number of price rows for a symbol."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM crypto_prices WHERE symbol = ?", symbol)
        return cursor.fetchone()[0]
