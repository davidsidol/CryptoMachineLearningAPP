"""
Data collection from CoinMarketCap and Coinbase APIs
"""
import requests
import time
from datetime import datetime, timedelta, timezone
import database as db
from config import (
    COINMARKETCAP_API_KEY, COINMARKETCAP_BASE_URL,
    COINBASE_API_KEY, COINBASE_BASE_URL,
    CRYPTO_ASSETS
)


# ---------------------------------------------------------------------------
# CoinMarketCap helpers
# ---------------------------------------------------------------------------

CMC_HEADERS = {
    "X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY,
    "Accept": "application/json",
}


def _cmc_get(endpoint: str, params: dict = None):
    url = f"{COINMARKETCAP_BASE_URL}{endpoint}"
    resp = requests.get(url, headers=CMC_HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_latest_quotes():
    """Fetch current price/volume/market-cap for all tracked symbols."""
    symbols = ",".join(CRYPTO_ASSETS.keys())
    data = _cmc_get("/v1/cryptocurrency/quotes/latest", {
        "symbol": symbols,
        "convert": "USD",
    })

    inserted = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for symbol, info in CRYPTO_ASSETS.items():
        entry = data.get("data", {}).get(symbol)
        if not entry:
            print(f"[CMC] No data for {symbol}")
            continue

        quote = entry["quote"]["USD"]
        try:
            db.upsert_price(
                symbol=symbol,
                timestamp=now,
                price=quote.get("price"),
                volume=quote.get("volume_24h"),
                market_cap=quote.get("market_cap"),
                pct_1h=quote.get("percent_change_1h"),
                pct_24h=quote.get("percent_change_24h"),
                pct_7d=quote.get("percent_change_7d"),
                circ_supply=entry.get("circulating_supply"),
                source="coinmarketcap",
            )
            inserted += 1
        except Exception as e:
            print(f"[CMC] DB error for {symbol}: {e}")

    db.log_collection("coinmarketcap_latest", "ALL", inserted, "success")
    print(f"[CMC] Latest quotes: {inserted} symbols inserted/updated.")
    return inserted


def fetch_historical_quotes(symbol: str, days: int = 365):
    """
    Pull daily historical data via CoinMarketCap v2 historical endpoint.
    Falls back to v1/cryptocurrency/quotes/historical.
    """
    cmc_id = CRYPTO_ASSETS[symbol]["cmc_id"]
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    print(f"[CMC] Fetching {days}d history for {symbol} (id={cmc_id})...")

    params = {
        "id": cmc_id,
        "time_start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": "daily",
        "convert": "USD",
        "count": min(days, 10000),
    }

    try:
        data = _cmc_get("/v2/cryptocurrency/quotes/historical", params)
        quotes = data.get("data", {}).get("quotes", [])
    except Exception as e:
        print(f"[CMC] v2 historical failed for {symbol}: {e}")
        return 0

    inserted = 0
    for q in quotes:
        ts_str = q.get("timestamp", "")
        quote = q.get("quote", {}).get("USD", {})
        if not ts_str or not quote:
            continue
        try:
            ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
            db.upsert_price(
                symbol=symbol,
                timestamp=ts,
                price=quote.get("price"),
                volume=quote.get("volume_24h"),
                market_cap=quote.get("market_cap"),
                pct_1h=quote.get("percent_change_1h"),
                pct_24h=quote.get("percent_change_24h"),
                pct_7d=quote.get("percent_change_7d"),
                circ_supply=q.get("circulating_supply"),
                source="coinmarketcap_historical",
            )
            inserted += 1
        except Exception as e:
            print(f"[CMC] Row error {symbol}: {e}")

    db.log_collection("coinmarketcap_historical", symbol, inserted, "success")
    print(f"[CMC] {symbol}: {inserted} historical rows inserted.")
    return inserted


# ---------------------------------------------------------------------------
# Coinbase (public REST API – no auth required for candles)
# ---------------------------------------------------------------------------

COINBASE_PUBLIC_URL = "https://api.exchange.coinbase.com"


def fetch_coinbase_candles(symbol: str, days: int = 300, granularity: int = 86400):
    """
    Fetch daily OHLCV candles from Coinbase Exchange public API.
    Coinbase limits to 300 candles per request; we paginate if needed.
    Skips symbols with no Coinbase listing (e.g. TAO).
    """
    product_id = CRYPTO_ASSETS[symbol].get("coinbase_id")
    if not product_id:
        print(f"[CB] {symbol} has no Coinbase listing — skipping candles.")
        return 0
    end_dt = datetime.now(timezone.utc)
    inserted = 0
    remaining = days

    print(f"[CB] Fetching candles for {symbol} ({product_id})...")

    while remaining > 0:
        batch = min(remaining, 300)
        start_dt = end_dt - timedelta(seconds=granularity * batch)

        params = {
            "granularity": granularity,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }

        try:
            url = f"{COINBASE_PUBLIC_URL}/products/{product_id}/candles"
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            print(f"[CB] Error fetching {symbol}: {e}")
            break

        if not candles:
            break

        for candle in candles:
            # [time, low, high, open, close, volume]
            try:
                ts = datetime.fromtimestamp(candle[0], tz=timezone.utc).replace(tzinfo=None)
                db.upsert_ohlcv(
                    symbol=symbol,
                    timestamp=ts,
                    open_p=candle[3],
                    high_p=candle[2],
                    low_p=candle[1],
                    close_p=candle[4],
                    volume=candle[5],
                    granularity=granularity,
                    source="coinbase",
                )
                inserted += 1
            except Exception as e:
                print(f"[CB] OHLCV row error: {e}")

        end_dt = start_dt
        remaining -= batch
        time.sleep(0.3)  # Rate limit courtesy

    db.log_collection("coinbase_candles", symbol, inserted, "success")
    print(f"[CB] {symbol}: {inserted} candles inserted.")
    return inserted


# ---------------------------------------------------------------------------
# Merge OHLCV into crypto_prices (fills gaps from CoinMarketCap)
# ---------------------------------------------------------------------------

def sync_ohlcv_to_prices(symbol: str):
    """Copy Coinbase close prices into crypto_prices where gaps exist."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO crypto_prices (symbol, timestamp, price_usd, volume_24h, source)
            SELECT o.symbol, o.timestamp, o.close_price, o.volume, 'coinbase_ohlcv'
            FROM crypto_ohlcv o
            WHERE o.symbol = ?
              AND o.granularity = 86400
              AND NOT EXISTS (
                  SELECT 1 FROM crypto_prices p
                  WHERE p.symbol = o.symbol
                    AND CAST(p.timestamp AS DATE) = CAST(o.timestamp AS DATE)
              )
        """, symbol)
        n = cursor.rowcount
        print(f"[SYNC] {symbol}: {n} OHLCV rows merged into crypto_prices.")


# ---------------------------------------------------------------------------
# Master bootstrap – run once to seed historical data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CoinGecko (free public API – no key needed for historical data)
# ---------------------------------------------------------------------------

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "TAO": "bittensor",
}
COINGECKO_URL = "https://api.coingecko.com/api/v3"


def fetch_coingecko_history(symbol: str, days: int = 365):
    """
    Fetch historical market data from CoinGecko free API.
    Returns daily price, volume, and market cap.
    """
    cg_id = COINGECKO_IDS.get(symbol)
    if not cg_id:
        return 0

    print(f"[CG] Fetching {days}d history for {symbol} from CoinGecko...")

    try:
        resp = requests.get(
            f"{COINGECKO_URL}/coins/{cg_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[CG] Error for {symbol}: {e}")
        return 0

    prices     = {int(p[0]): p[1] for p in data.get("prices", [])}
    volumes    = {int(v[0]): v[1] for v in data.get("total_volumes", [])}
    mcaps      = {int(m[0]): m[1] for m in data.get("market_caps", [])}

    inserted = 0
    for ts_ms, price in prices.items():
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(tzinfo=None)
        # Snap to midnight
        ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        vol  = volumes.get(ts_ms)
        mcap = mcaps.get(ts_ms)
        try:
            db.upsert_price(
                symbol=symbol,
                timestamp=ts,
                price=price,
                volume=vol,
                market_cap=mcap,
                source="coingecko",
            )
            inserted += 1
        except Exception as e:
            print(f"[CG] Row error {symbol} {ts}: {e}")

    db.log_collection("coingecko_historical", symbol, inserted, "success")
    print(f"[CG] {symbol}: {inserted} rows inserted/updated.")
    return inserted


def enrich_market_cap_from_coingecko():
    """Back-fill missing market_cap in crypto_prices using CoinGecko."""
    print("[CG] Back-filling market cap data...")
    for symbol in CRYPTO_ASSETS:
        fetch_coingecko_history(symbol, days=365)
        time.sleep(1.5)  # CoinGecko rate limit: ~10-30 req/min


def bootstrap_all(days: int = 365):
    """Pull full historical data for all symbols."""
    print("=" * 60)
    print(f"[BOOTSTRAP] Seeding {days} days of historical data...")
    print("=" * 60)

    for symbol in CRYPTO_ASSETS:
        print(f"\n--- {symbol} ---")
        # 1. CoinMarketCap historical (primary – price + volume + market cap)
        n = fetch_historical_quotes(symbol, days=days)
        time.sleep(1)

        # 2. Coinbase OHLCV (adds OHLC, fills any CMC gaps)
        fetch_coinbase_candles(symbol, days=min(days, 300))
        time.sleep(0.5)

        # 3. Merge OHLCV rows into prices table
        sync_ohlcv_to_prices(symbol)

        # 4. CoinGecko fills market cap + volume gaps (free)
        fetch_coingecko_history(symbol, days=days)
        time.sleep(1.5)

    # 5. Grab current snapshot from CMC
    print("\n[CMC] Fetching latest live quotes...")
    fetch_latest_quotes()

    print("\n[BOOTSTRAP] Done.")


def refresh_all():
    """Pull latest quotes (called by scheduler)."""
    print("[REFRESH] Fetching latest quotes...")
    fetch_latest_quotes()
    for symbol in CRYPTO_ASSETS:
        fetch_coinbase_candles(symbol, days=2)  # last 2 days
        sync_ohlcv_to_prices(symbol)
        fetch_coingecko_history(symbol, days=3)
        time.sleep(1)


if __name__ == "__main__":
    bootstrap_all(days=365)
