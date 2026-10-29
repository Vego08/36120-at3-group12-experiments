"""Engineered features defined in the 36120 AT3 brief.

All rolling windows are computed per token and use only information up to day t,
so no feature looks into the future. Rolling statistics need a full window, so the
first rows of each token are NaN (19 rows for 20-day windows, 20 for Return20 and
Volatility20).

Ownership (from the brief):
    Student A: DailyReturn, Return5, Return20, Momentum
    Student B: MA5, MA20, TrendStrength
    Student C: Volatility20, MaxPrice20, MaxDrawdown, GapRisk
    Student D: VolumeSMA20, RelativeVolume

Every feature is implemented here so the experiment notebook is self-contained.
In production, the Student D API serves VolumeSMA20 and RelativeVolume and fetches
the rest from the other students' APIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SHORT = 5
WINDOW_LONG = 20

STUDENT_D_FEATURES = ["VolumeSMA20", "RelativeVolume"]
TOR_MODEL_FEATURES = ["Return20", "RelativeVolume", "Volatility20", "MA5", "MA20"]


def _by_token(df: pd.DataFrame, col: str):
    return df.groupby("token", sort=False)[col]


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Student A features."""
    out = df.copy()
    close = _by_token(out, "close")
    out["DailyReturn"] = close.pct_change(1, fill_method=None)
    out["Return5"] = close.pct_change(WINDOW_SHORT, fill_method=None)
    out["Return20"] = close.pct_change(WINDOW_LONG, fill_method=None)
    out["Momentum"] = out["close"] - close.shift(WINDOW_LONG)
    return out


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Student B features."""
    out = df.copy()
    close = _by_token(out, "close")
    out["MA5"] = close.transform(lambda s: s.rolling(WINDOW_SHORT).mean())
    out["MA20"] = close.transform(lambda s: s.rolling(WINDOW_LONG).mean())
    out["TrendStrength"] = out["MA5"] / out["MA20"] - 1
    return out


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Student C features. Requires DailyReturn (call add_return_features first).

    Volatility20 uses the pandas default sample standard deviation (ddof=1).
    GapRisk: the brief gives no window for the mean, so a 20-day window is assumed.
    """
    out = df.copy()
    out["Volatility20"] = _by_token(out, "DailyReturn").transform(
        lambda s: s.rolling(WINDOW_LONG).std()
    )
    out["MaxPrice20"] = _by_token(out, "high").transform(lambda s: s.rolling(WINDOW_LONG).max())
    out["MaxDrawdown"] = (out["MaxPrice20"] - out["close"]) / out["MaxPrice20"]

    prev_close = _by_token(out, "close").shift(1)
    out["_gap"] = (out["open"] - prev_close).abs() / prev_close
    out["GapRisk"] = _by_token(out, "_gap").transform(lambda s: s.rolling(WINDOW_LONG).mean())
    return out.drop(columns="_gap")


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Student D features (owned by this service).

    VolumeSMA20     = mean(Volume_t, ..., Volume_t-19)
    RelativeVolume  = Volume_t / VolumeSMA20
    """
    out = df.copy()
    out["VolumeSMA20"] = _by_token(out, "volume").transform(lambda s: s.rolling(WINDOW_LONG).mean())
    out["RelativeVolume"] = out["volume"] / out["VolumeSMA20"].replace(0, np.nan)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered feature from the brief. Input needs token, date and OHLCV."""
    required = {"token", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.sort_values(["token", "date"]).reset_index(drop=True)
    out = add_return_features(out)
    out = add_moving_average_features(out)
    out = add_risk_features(out)
    out = add_volume_features(out)
    return out


# ---------------------------------------------------------------------------
# Engineered features for the TOR model (section F of the experiment notebook)
# ---------------------------------------------------------------------------
TOR_ENGINEERED_FEATURES = ["MA_ratio", "OS_t", "Return13", "OS_proj"]


def projected_opportunity_score(df: pd.DataFrame, horizon: int = 7) -> pd.Series:
    """Opportunity Score at t+horizon, assuming price stays flat and volume stays at its 20-day average.

    Every 20-day window of day t+7 shares its first 13 days (t-12..t) with information already known
    on day t. This projection keeps that known part exactly and fills the 7 unknown days with a
    'nothing changes' assumption, giving the model a leakage-free estimate of next week's score.
    """
    n_known = WINDOW_LONG - horizon  # 13 known days inside the future 20-day window
    out = df.sort_values(["token", "date"])
    g = out.groupby("token", sort=False)

    close = out["close"]
    ret_proj = close / g["close"].shift(n_known) - 1                     # Return20 at t+7
    ma20_proj = (g["close"].transform(lambda s: s.rolling(n_known).sum()) + horizon * close) / WINDOW_LONG
    trend_proj = close / ma20_proj                                        # MA5/MA20 at t+7

    dr = g["close"].pct_change(1, fill_method=None)
    s1 = dr.groupby(out["token"], sort=False).transform(lambda s: s.rolling(n_known).sum())
    s2 = (dr ** 2).groupby(out["token"], sort=False).transform(lambda s: s.rolling(n_known).sum())
    mean = s1 / WINDOW_LONG                                               # 7 future returns are 0
    var = (s2 - WINDOW_LONG * mean ** 2) / (WINDOW_LONG - 1)
    vol_proj = np.sqrt(var.clip(lower=0))                                 # Volatility20 at t+7

    vol_sma = g["volume"].transform(lambda s: s.rolling(WINDOW_LONG).mean())
    vol_known = g["volume"].transform(lambda s: s.rolling(n_known).sum())
    relvol_proj = vol_sma / ((vol_known + horizon * vol_sma) / WINDOW_LONG)  # RelativeVolume at t+7

    os_proj = (0.40 * (ret_proj + 0.20) / 0.40
               + 0.30 * trend_proj
               + 0.20 * relvol_proj.clip(upper=2) / 2
               - 0.10 * (vol_proj / 0.10).clip(upper=1))
    return os_proj.reindex(df.index)


def add_tor_engineered_features(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Add MA_ratio, OS_t, Return13 and OS_proj. Needs build_features() output (incl. close, volume)."""
    from at3_crypto.target import opportunity_score  # local import avoids a circular import

    out = df.copy()
    out["MA_ratio"] = out["MA5"] / out["MA20"]
    out["OS_t"] = opportunity_score(out)
    out["Return13"] = _by_token(out, "close").pct_change(WINDOW_LONG - horizon, fill_method=None)
    out["OS_proj"] = projected_opportunity_score(out, horizon=horizon)
    return out

