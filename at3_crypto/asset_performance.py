"""Asset Performance Index (API): Student A's target, as defined in the 36120 AT3 brief.

The index is a 0-100 score that summarises how strongly a cryptocurrency is performing.
It combines four ingredients, each first converted to a 0-1 score:

    RS = min(max((Return20 + 0.20) / 0.40, 0), 1)        Return Strength       (weight 0.40)
    TS = min(max((TrendStrength + 0.10) / 0.20, 0), 1)   Trend Strength Score  (weight 0.30)
    RV = min(RelativeVolume, 2) / 2                      Relative Volume Score (weight 0.20)
    PS = 1 - Volatility20                                Price Stability       (weight 0.10)

    API = 100 * (0.40*RS + 0.30*TS + 0.20*RV + 0.10*PS)

Forecasting set-up (brief: predict t+7):
    API_t       the index on day t (uses information up to day t only)
    API_target  the index on day t+7, i.e. API_t shifted back 7 rows WITHIN each token.
                This is the value the model learns to predict from day-t information.

Keeping these functions in the shared package (instead of only in the notebook) means the
notebook and the FastAPI service compute the index in exactly the same way.
"""

from __future__ import annotations

import pandas as pd

# Brief: every model forecasts 7 days ahead
FORECAST_HORIZON = 7

# The four features the brief requires for the API model (names match the shared feature code)
API_INPUT_FEATURES = ["Return20", "TrendStrength", "RelativeVolume", "Volatility20"]

# Weight of each 0-1 component in the final index (they add up to 1.0, so API is out of 100)
API_WEIGHTS = {"RS": 0.40, "TS": 0.30, "RV": 0.20, "PS": 0.10}


def api_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return the four intermediate 0-1 scores (RS, TS, RV, PS) for every row.

    Needs the columns Return20, TrendStrength, RelativeVolume and Volatility20.
    A missing input (NaN) gives a missing component, so the index is never computed
    from incomplete information.
    """
    missing = set(API_INPUT_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns for the API index: {sorted(missing)}")

    return pd.DataFrame(
        {
            # 20-day return mapped from [-20%, +20%] to [0, 1]; anything beyond is capped
            "RS": ((df["Return20"] + 0.20) / 0.40).clip(lower=0, upper=1),
            # MA5/MA20 - 1 mapped from [-10%, +10%] to [0, 1]; anything beyond is capped
            "TS": ((df["TrendStrength"] + 0.10) / 0.20).clip(lower=0, upper=1),
            # Volume up to twice its 20-day average earns credit; above 2x is capped at 1
            "RV": df["RelativeVolume"].clip(upper=2) / 2,
            # Lower volatility = more stable price = higher score (not clipped, as in the brief)
            "PS": 1 - df["Volatility20"],
        },
        index=df.index,
    )


def asset_performance_index(df: pd.DataFrame) -> pd.Series:
    """Compute the Asset Performance Index (0-100) for every row."""
    comp = api_components(df)
    weighted_sum = sum(weight * comp[name] for name, weight in API_WEIGHTS.items())
    return (100 * weighted_sum).rename("API")


def add_api_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Add today's index, its components and the t+horizon target. Needs build_features() output.

    Adds:
        RS, TS, RV, PS   the four components on day t
        API_t            the index on day t (known today)
        API_target       the index on day t+horizon (the value to predict)
        target_date      the calendar date the target refers to (date + horizon days)

    The shift is done per token, so one coin's future is never used as another coin's label.
    The last `horizon` rows of each token have no future value yet, so API_target is NaN there.
    """
    out = df.sort_values(["token", "date"]).reset_index(drop=True)
    out[list(API_WEIGHTS)] = api_components(out)
    out["API_t"] = asset_performance_index(out)
    out["API_target"] = out.groupby("token", sort=False)["API_t"].shift(-horizon)
    out["target_date"] = out["date"] + pd.Timedelta(days=horizon)
    return out