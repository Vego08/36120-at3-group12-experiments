"""Data acquisition: daily OHLCV candles from the Kraken public REST API.

Docs: https://docs.kraken.com/api/docs/rest-api/get-ohlc-data

Key facts about the endpoint that this module handles:
* Public endpoint, no API key required.
* interval=1440 returns daily candles.
* Only the latest 720 candles are available.
* The last candle is the current, unfinished day, so it is dropped.
* Prices and volume are returned as strings and are converted to floats.
* Volume is in base-currency units (e.g. BTC), not USD.

Run from the project root to pull and save a fresh snapshot:
    python -m at3_crypto.dataset
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

from loguru import logger
import pandas as pd
import requests

from at3_crypto.config import RAW_DATA_DIR

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
DAILY_INTERVAL = 1440  # minutes

# Kraken uses XBT for Bitcoin and XDG for Dogecoin
TOKEN_TO_PAIR: dict[str, str] = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "DOGE": "XDGUSD",
    "XMR": "XMRUSD",
    "XLM": "XLMUSD",
    "TRX": "TRXUSD",
}
SUPPORTED_TOKENS: list[str] = list(TOKEN_TO_PAIR)

KRAKEN_COLUMNS = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
OUTPUT_COLUMNS = ["token", "date", "open", "high", "low", "close", "vwap", "volume", "count"]
SNAPSHOT_PREFIX = "kraken_ohlc_daily_"


class KrakenAPIError(RuntimeError):
    """Raised when Kraken returns an error message or an unexpected payload."""


def _parse_payload(payload: dict, token: str) -> pd.DataFrame:
    """Convert a raw Kraken OHLC JSON payload into a clean daily DataFrame."""
    if payload.get("error"):
        raise KrakenAPIError(f"{token}: Kraken returned {payload['error']}")

    result = payload.get("result") or {}
    pair_keys = [k for k in result if k != "last"]
    if not pair_keys:
        raise KrakenAPIError(f"{token}: no OHLC data in response")

    df = pd.DataFrame(result[pair_keys[0]], columns=KRAKEN_COLUMNS)
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")

    # Kraken timestamps are the UTC start of each daily candle
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()

    # Drop the current, unfinished day (and anything after it)
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["date"] < today_utc]

    df["token"] = token
    return df[OUTPUT_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_ohlc(
    token: str,
    interval: int = DAILY_INTERVAL,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Fetch completed daily OHLCV candles for one token from Kraken.

    Retries with exponential backoff on network errors and HTTP errors.
    Raises ValueError for unsupported tokens and KrakenAPIError if all attempts fail.
    """
    token = token.upper().strip()
    if token not in TOKEN_TO_PAIR:
        raise ValueError(f"Unsupported token '{token}'. Supported: {SUPPORTED_TOKENS}")

    params = {"pair": TOKEN_TO_PAIR[token], "interval": interval}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(KRAKEN_OHLC_URL, params=params, timeout=timeout)
            response.raise_for_status()
            return _parse_payload(response.json(), token)
        except (requests.RequestException, ValueError, KrakenAPIError) as exc:
            last_error = exc
            logger.warning(f"{token}: attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(backoff ** (attempt - 1))

    raise KrakenAPIError(f"{token}: all {retries} attempts failed. Last error: {last_error}")


def fetch_all(
    tokens: list[str] | None = None, pause: float = 1.0
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch every token, pausing between calls to respect Kraken rate limits.

    Returns the combined DataFrame and a dict of {token: error message} for failures,
    so one unavailable token does not stop the others.
    """
    tokens = tokens or SUPPORTED_TOKENS
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for i, token in enumerate(tokens):
        try:
            df = fetch_ohlc(token)
            frames.append(df)
            logger.info(
                f"{token}: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}"
            )
        except (KrakenAPIError, ValueError) as exc:
            failures[token] = str(exc)
            logger.error(str(exc))
        if i < len(tokens) - 1:
            time.sleep(pause)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return combined, failures


def save_snapshot(df: pd.DataFrame, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Save a dated CSV snapshot so experiments are reproducible."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = raw_dir / f"{SNAPSHOT_PREFIX}{stamp}.csv"
    df.to_csv(path, index=False)
    logger.success(f"Saved {len(df)} rows to {path}")
    return path


def load_latest_snapshot(raw_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, Path]:
    """Load the most recent saved snapshot from data/raw."""
    snapshots = sorted(raw_dir.glob(f"{SNAPSHOT_PREFIX}*.csv"))
    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot found in {raw_dir}. Run `python -m at3_crypto.dataset` first."
        )
    path = snapshots[-1]
    df = pd.read_csv(path, parse_dates=["date"])
    return df, path


if __name__ == "__main__":
    data, failed = fetch_all()
    if not data.empty:
        save_snapshot(data)
    if failed:
        logger.error(f"Failed tokens: {list(failed)}")