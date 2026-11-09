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

        # Prediction outcomes — actual vs predicted once the date arrives
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='prediction_outcomes' AND xtype='U')
            CREATE TABLE prediction_outcomes (
                id                BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol            NVARCHAR(10)  NOT NULL,
                target            NVARCHAR(50)  NOT NULL,
                horizon_days      INT           NOT NULL,
                predicted_for     DATE          NOT NULL,
                predicted_value   FLOAT         NOT NULL,
                actual_value      FLOAT         NOT NULL,
                abs_error         FLOAT         NOT NULL,
                pct_error         FLOAT         NOT NULL,
                direction_correct BIT           NOT NULL,
                evaluated_at      DATETIME2     DEFAULT GETUTCDATE(),
                CONSTRAINT UQ_outcome UNIQUE (symbol, target, horizon_days, predicted_for)
            )
        """)

        # v2 model metrics (out-of-sample, vs "no change" baseline)
        for col, typ in [("baseline_mae", "FLOAT"), ("skill_score", "FLOAT"),
                         ("directional_acc", "FLOAT"), ("up_share", "FLOAT"),
                         ("model_version", "INT")]:
            cursor.execute(f"IF COL_LENGTH('ml_models', '{col}') IS NULL "
                           f"ALTER TABLE ml_models ADD {col} {typ} NULL")

        # Crypto Fear & Greed Index (alternative.me), one row per day
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='fear_greed' AND xtype='U')
            CREATE TABLE fear_greed (
                [date]          DATE          NOT NULL PRIMARY KEY,
                value           INT           NOT NULL,
                classification  NVARCHAR(30)  NOT NULL,
                updated_at      DATETIME2     DEFAULT GETUTCDATE()
            )
        """)

        # Perp funding (Hyperliquid), aggregated to one row per day
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='funding_daily' AND xtype='U')
            CREATE TABLE funding_daily (
                symbol       NVARCHAR(10) NOT NULL,
                [date]       DATE         NOT NULL,
                funding_avg  FLOAT        NOT NULL,   -- mean hourly funding rate
                premium_avg  FLOAT,
                samples      INT,
                CONSTRAINT PK_funding_daily PRIMARY KEY (symbol, [date])
            )
        """)

        # Hourly perp market snapshot (funding, open interest, leverage)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='futures_snapshot' AND xtype='U')
            CREATE TABLE futures_snapshot (
                id             BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol         NVARCHAR(10) NOT NULL,
                ts             DATETIME2    NOT NULL DEFAULT GETUTCDATE(),
                venue          NVARCHAR(20) NOT NULL DEFAULT 'hyperliquid',
                funding        FLOAT,
                open_interest  FLOAT,       -- in coins
                oi_usd         FLOAT,
                mark_px        FLOAT,
                premium        FLOAT,
                day_volume_usd FLOAT,
                max_leverage   INT
            )
        """)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='IX_futures_snapshot_sym_ts')
            CREATE INDEX IX_futures_snapshot_sym_ts ON futures_snapshot (symbol, ts)
        """)

        # Hourly options snapshot (Deribit, BTC/ETH)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='options_snapshot' AND xtype='U')
            CREATE TABLE options_snapshot (
                id              BIGINT IDENTITY(1,1) PRIMARY KEY,
                currency        NVARCHAR(10) NOT NULL,
                ts              DATETIME2    NOT NULL DEFAULT GETUTCDATE(),
                call_oi         FLOAT,
                put_oi          FLOAT,
                put_call_ratio  FLOAT,
                oi_usd          FLOAT,
                dvol            FLOAT,
                underlying_px   FLOAT
            )
        """)

        # Hidden live test of the 1-day up/down classifier (not shown on the dashboard)
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='direction_shadow' AND xtype='U')
            CREATE TABLE direction_shadow (
                id            BIGINT IDENTITY(1,1) PRIMARY KEY,
                symbol        NVARCHAR(10) NOT NULL,
                horizon_days  INT          NOT NULL,
                generated_at  DATETIME2    NOT NULL DEFAULT GETUTCDATE(),
                target_date   DATE         NOT NULL,
                base_price    FLOAT        NOT NULL,
                prob_up       FLOAT        NOT NULL,
                actual_close  FLOAT        NULL,
                correct       BIT          NULL,
                scored_at     DATETIME2    NULL
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


def get_unevaluated_predictions() -> pd.DataFrame:
    """
    Return the latest prediction for each (symbol, target, horizon, predicted_for)
    where predicted_for has passed and no outcome has been recorded yet.
    """
    with get_connection() as conn:
        query = """
            WITH latest AS (
                SELECT symbol, target, horizon_days,
                       CAST(predicted_for AS DATE) AS predicted_for,
                       predicted_value, generated_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, target, horizon_days, CAST(predicted_for AS DATE)
                           ORDER BY generated_at DESC
                       ) AS rn
                FROM predictions
                -- only once the whole target day is over (forecasts target the day's close)
                WHERE CAST(predicted_for AS DATE) < CAST(GETUTCDATE() AS DATE)
            )
            SELECT l.symbol, l.target, l.horizon_days, l.predicted_for, l.predicted_value, l.generated_at
            FROM latest l
            WHERE l.rn = 1
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_outcomes o
                  WHERE o.symbol = l.symbol
                    AND o.target = l.target
                    AND o.horizon_days = l.horizon_days
                    AND o.predicted_for = l.predicted_for
              )
        """
        return pd.read_sql(query, conn, parse_dates=["predicted_for"])


def save_prediction_outcome(symbol: str, target: str, horizon: int,
                             predicted_for, predicted_value: float,
                             actual_value: float, abs_error: float,
                             pct_error: float, direction_correct: bool):
    """Record how accurate a past prediction turned out to be."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            MERGE prediction_outcomes AS tgt
            USING (SELECT ? AS symbol, ? AS target, ? AS horizon_days, CAST(? AS DATE) AS predicted_for) AS src
            ON tgt.symbol = src.symbol AND tgt.target = src.target
               AND tgt.horizon_days = src.horizon_days AND tgt.predicted_for = src.predicted_for
            WHEN NOT MATCHED THEN
                INSERT (symbol, target, horizon_days, predicted_for, predicted_value,
                        actual_value, abs_error, pct_error, direction_correct)
                VALUES (?, ?, ?, CAST(? AS DATE), ?, ?, ?, ?, ?);
        """, symbol, target, horizon, str(predicted_for)[:10],
             symbol, target, horizon, str(predicted_for)[:10],
             predicted_value, actual_value, abs_error, pct_error,
             1 if direction_correct else 0)


def get_prediction_accuracy() -> pd.DataFrame:
    """
    Return per-model accuracy stats from evaluated predictions.
    Columns: symbol, target, horizon_days, sample_count,
             directional_accuracy (%), mean_abs_pct_error (%)
    """
    with get_connection() as conn:
        query = """
            SELECT
                symbol,
                target,
                horizon_days,
                COUNT(*)                                                    AS sample_count,
                SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS directional_accuracy,
                AVG(ABS(pct_error))                                         AS mean_abs_pct_error
            FROM prediction_outcomes
            GROUP BY symbol, target, horizon_days
            ORDER BY symbol, target, horizon_days
        """
        return pd.read_sql(query, conn)


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

def upsert_fear_greed(day, value: int, classification: str):
    with get_connection() as conn:
        conn.cursor().execute("""
            MERGE fear_greed AS t
            USING (SELECT ? AS d, ? AS v, ? AS c) AS s
            ON t.[date] = s.d
            WHEN MATCHED THEN UPDATE SET value = s.v, classification = s.c, updated_at = GETUTCDATE()
            WHEN NOT MATCHED THEN INSERT ([date], value, classification) VALUES (s.d, s.v, s.c);
        """, day, value, classification)


def get_fear_greed_count() -> int:
    with get_connection() as conn:
        return conn.cursor().execute("SELECT COUNT(*) FROM fear_greed").fetchone()[0]


def get_fear_greed(days: int = 31) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT TOP (?) [date], value, classification
            FROM fear_greed ORDER BY [date] DESC
        """, conn, params=[days])


# ---------------------------------------------------------------------------
# Futures / options market context
# ---------------------------------------------------------------------------

def upsert_funding_daily(symbol: str, day, funding_avg: float, premium_avg, samples: int):
    with get_connection() as conn:
        conn.cursor().execute("""
            MERGE funding_daily AS t
            USING (SELECT ? AS s, ? AS d) AS k ON t.symbol = k.s AND t.[date] = k.d
            WHEN MATCHED THEN UPDATE SET funding_avg = ?, premium_avg = ?, samples = ?
            WHEN NOT MATCHED THEN INSERT (symbol, [date], funding_avg, premium_avg, samples)
                 VALUES (?, ?, ?, ?, ?);
        """, symbol, day, funding_avg, premium_avg, samples,
             symbol, day, funding_avg, premium_avg, samples)


def get_funding_count(symbol: str) -> int:
    with get_connection() as conn:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM funding_daily WHERE symbol = ?", symbol).fetchone()[0]


def get_funding_daily(symbol: str, days: int = 730) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT [date], funding_avg, premium_avg FROM funding_daily
            WHERE symbol = ? AND [date] >= DATEADD(day, -?, CAST(GETUTCDATE() AS DATE))
            ORDER BY [date]
        """, conn, params=[symbol, days])


def insert_futures_snapshot(rows: list):
    with get_connection() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute("""
                INSERT INTO futures_snapshot
                    (symbol, venue, funding, open_interest, oi_usd, mark_px, premium,
                     day_volume_usd, max_leverage)
                VALUES (?, 'hyperliquid', ?, ?, ?, ?, ?, ?, ?)
            """, r["symbol"], r["funding"], r["open_interest"], r["oi_usd"], r["mark_px"],
                 r["premium"], r["day_volume_usd"], r["max_leverage"])


def insert_options_snapshot(rows: list):
    with get_connection() as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute("""
                INSERT INTO options_snapshot
                    (currency, call_oi, put_oi, put_call_ratio, oi_usd, dvol, underlying_px)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, r["currency"], r["call_oi"], r["put_oi"], r["put_call_ratio"],
                 r["oi_usd"], r["dvol"], r["underlying_px"])


def get_futures_latest() -> pd.DataFrame:
    """Latest snapshot per symbol plus the snapshot closest to 24h earlier."""
    with get_connection() as conn:
        return pd.read_sql("""
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) rn
                FROM futures_snapshot
            ),
            prev AS (
                SELECT symbol, oi_usd, funding,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) rn
                FROM futures_snapshot
                WHERE ts <= DATEADD(hour, -23, GETUTCDATE())
            )
            SELECT l.symbol, l.ts, l.funding, l.open_interest, l.oi_usd, l.mark_px,
                   l.premium, l.day_volume_usd, l.max_leverage,
                   p.oi_usd AS oi_usd_24h_ago, p.funding AS funding_24h_ago
            FROM latest l LEFT JOIN prev p ON p.symbol = l.symbol AND p.rn = 1
            WHERE l.rn = 1
        """, conn)


def get_options_latest() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            WITH x AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY currency ORDER BY ts DESC) rn
                FROM options_snapshot
            )
            SELECT currency, ts, call_oi, put_oi, put_call_ratio, oi_usd, dvol, underlying_px
            FROM x WHERE rn = 1
        """, conn)


def delete_outcomes_since(day) -> int:
    """Remove scored outcomes from `day` on (used to re-score with corrected logic)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM prediction_outcomes WHERE predicted_for >= ?", day)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Hidden up/down classifier test
# ---------------------------------------------------------------------------

def insert_shadow(symbol, horizon, target_date, base_price, prob_up):
    with get_connection() as conn:
        conn.cursor().execute("""
            INSERT INTO direction_shadow (symbol, horizon_days, target_date, base_price, prob_up)
            VALUES (?, ?, ?, ?, ?)
        """, symbol, horizon, target_date, base_price, prob_up)


def get_unscored_shadow() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("""
            SELECT id, symbol, horizon_days, generated_at, target_date, base_price, prob_up
            FROM direction_shadow
            WHERE correct IS NULL AND target_date < CAST(GETUTCDATE() AS DATE)
        """, conn)


def score_shadow(row_id, actual_close, correct):
    with get_connection() as conn:
        conn.cursor().execute("""
            UPDATE direction_shadow SET actual_close = ?, correct = ?, scored_at = GETUTCDATE()
            WHERE id = ?
        """, actual_close, correct, row_id)
