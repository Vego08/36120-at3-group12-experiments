"""Trading Opportunity Rating (TOR) target, as defined in the 36120 AT3 brief.

Intermediate variables:
    M = (Return20 + 0.20) / 0.40               Momentum Score (not clipped, per the brief)
    T = MA5 / MA20                             Trend Score
    V = min(RelativeVolume, 2) / 2             Volume Score
    R = min(Volatility20 / 0.10, 1)            Volatility Penalty

Opportunity Score:
    OS = 0.40*M + 0.30*T + 0.20*V - 0.10*R

Classes:
    OS < 0.20          Strong Sell
    0.20 <= OS < 0.40  Sell
    0.40 <= OS < 0.60  Hold
    0.60 <= OS < 0.80  Buy
    OS >= 0.80         Strong Buy

Forecasting set-up: the label for row t is the class computed from day t+7's values,
while the features of row t only use information up to day t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_HORIZON = 7

TOR_LABELS: list[str] = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(TOR_LABELS)}
ID_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_ID.items()}
OS_BINS = [-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf]


def opportunity_score(df: pd.DataFrame) -> pd.Series:
    """OS from Return20, MA5, MA20, RelativeVolume and Volatility20 (NaN if any input is NaN)."""
    m = (df["Return20"] + 0.20) / 0.40
    t = df["MA5"] / df["MA20"]
    v = df["RelativeVolume"].clip(upper=2) / 2
    r = (df["Volatility20"] / 0.10).clip(upper=1)
    return 0.40 * m + 0.30 * t + 0.20 * v - 0.10 * r


def score_to_class_id(score: pd.Series) -> pd.Series:
    """Bin OS into class ids 0..4 (left-closed bins, matching the brief's <= / < rules)."""
    ids = pd.cut(score, bins=OS_BINS, labels=False, right=False)
    return ids.astype("Int64")


def add_tor_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Add current and future OS/class columns. Needs features from build_features().

    Adds:
        OS_t           Opportunity Score on day t (known today)
        TOR_t          class id on day t (used only for the persistence baseline)
        OS_future      Opportunity Score on day t+horizon
        TOR            target class id on day t+horizon (0..4)
        TOR_label      target class name
    """
    out = df.sort_values(["token", "date"]).reset_index(drop=True)
    out["OS_t"] = opportunity_score(out)
    out["TOR_t"] = score_to_class_id(out["OS_t"])

    grouped = out.groupby("token", sort=False)
    out["OS_future"] = grouped["OS_t"].shift(-horizon)
    out["TOR"] = grouped["TOR_t"].shift(-horizon).astype("Int64")
    out["TOR_label"] = out["TOR"].map(ID_TO_LABEL)
    return out