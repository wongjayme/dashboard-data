"""
Build stock dashboard data for static GitHub Pages deployment.

Flow:
  1. Download fresh stock universe CSV (RS-ranked list)
  2. Filter to AvgVol10 >= 100k
  3. Batch-download 1y price history from yfinance in parallel
  4. Compute metrics per stock
  5. Write data/snapshot.json + data/meta.json

Run from repo root:
  python scripts/build_data.py [--out-dir data] [--csv-url URL] [--workers 10]

GitHub Actions:
  python scripts/build_data.py --out-dir data --csv-url "$CSV_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CSV_URL     = os.environ.get("CSV_URL", "")
INDUSTRIES_CSV_URL  = os.environ.get("INDUSTRIES_CSV_URL", "")
MIN_AVG_VOLUME  = 90_000     # AvgVol50 filter — anything below gets dropped
MIN_PRICE       = 1.0        # last close must be >= $1
MIN_MARKET_CAP  = 100_000_000  # MarketCap filter — anything below $100M gets dropped

# Stale / halted ticker detection — a ticker whose most recent bar is this many
# calendar days older than SPY's most recent bar is treated as not currently
# trading (halt, delisting, feed issue, etc.) and dropped. 4 calendar days
# covers a normal weekend gap plus a 1-day feed delay without much risk of
# false-flagging a ticker that's actually still trading normally.
STALE_MAX_CALENDAR_DAYS = 4

# Industry trend sparkline — trailing window + sample count
SPARK_WINDOW = 63   # trading days ≈ 3 months, matches the existing `3m` field
SPARK_POINTS = 13   # evenly-spaced samples across the window (~weekly cadence)

# Ticker suffixes that indicate ETFs/funds — drop these
ETF_SUFFIXES = ()  # yfinance universe shouldn't have ETFs, but just in case

# Columns to pass through from CSV into snapshot as-is (if present)
CSV_PASSTHROUGH = [
    "Rank", "Percentile",
    "MarketCap", "PctFrom52WkHigh", "AvgVol50",
]

# MA combos for Dist/MA column (matches ETF dashboard)
DIST_MA_COMBOS = [
    ("SMA", 5), ("SMA", 8), ("SMA", 10), ("SMA", 21), ("SMA", 50), ("SMA", 65), ("SMA", 150), ("SMA", 200),
    ("EMA", 5), ("EMA", 8), ("EMA", 10), ("EMA", 21), ("EMA", 50), ("EMA", 65), ("EMA", 150), ("EMA", 200),
]

# ---------------------------------------------------------------------------
# Supplemental tickers — always included regardless of CSV contents.
# Add any ticker missing from your nightly CSV feed here.
# Required fields: Ticker, Sector, Industry
# Optional: any CSV_PASSTHROUGH column (leave blank to use None)
# ---------------------------------------------------------------------------
_SUPPLEMENTAL_CSV = os.path.join(os.path.dirname(__file__), 'supplemental.csv')
SUPPLEMENTAL_TICKERS = (
    pd.read_csv(_SUPPLEMENTAL_CSV).to_dict('records')
    if os.path.exists(_SUPPLEMENTAL_CSV) else []
)

# ---------------------------------------------------------------------------
# Industry overrides — reclassify specific tickers into custom industries.
# Tickers must already exist in the CSV feed (or supplemental.csv).
# Required columns: Ticker, Industry, Sector
# ---------------------------------------------------------------------------
_INDUSTRY_OVERRIDES_CSV = os.path.join(os.path.dirname(__file__), 'industry_overrides.csv')
INDUSTRY_OVERRIDES: dict[str, dict] = {}
if os.path.exists(_INDUSTRY_OVERRIDES_CSV):
    _ov_df = pd.read_csv(_INDUSTRY_OVERRIDES_CSV)
    for _, _row in _ov_df.iterrows():
        ticker = str(_row["Ticker"]).strip()
        if ticker:
            INDUSTRY_OVERRIDES[ticker] = {
                "industry": str(_row["Industry"]).strip(),
                "sector":   str(_row["Sector"]).strip(),
            }
    print(f"Loaded {len(INDUSTRY_OVERRIDES)} industry overrides from industry_overrides.csv")

# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

def load_universe(csv_url: str) -> pd.DataFrame:
    """Download and validate the stock universe CSV."""
    print(f"Fetching universe CSV: {csv_url}")
    df = pd.read_csv(csv_url)

    required = {"Ticker", "Sector", "Industry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    total_before = len(df)

    if "AvgVol50" in df.columns:
        df["AvgVol50"] = pd.to_numeric(df["AvgVol50"], errors="coerce")
        df = df[df["AvgVol50"] >= MIN_AVG_VOLUME].copy()
        removed = total_before - len(df)
        print(f"Volume filter (AvgVol50 >= {MIN_AVG_VOLUME:,}): "
              f"{total_before} → {len(df)} stocks ({removed} removed)")
    else:
        print("Warning: AvgVol50 column not found, skipping volume filter")

    if "MarketCap" in df.columns:
        before = len(df)
        df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
        df = df[df["MarketCap"] >= MIN_MARKET_CAP].copy()
        print(f"MarketCap filter (>= ${MIN_MARKET_CAP/1e6:.0f}M): "
              f"{before} → {len(df)} stocks ({before - len(df)} removed)")
    else:
        print("Warning: MarketCap column not found, skipping market cap filter")

    # Drop ETFs — identified by Exchange value or common ETF ticker patterns
    if "Exchange" in df.columns:
        before = len(df)
        etf_mask = df["Exchange"].astype(str).str.upper().isin(["ETF", "FUND"])
        df = df[~etf_mask].copy()
        removed = before - len(df)
        if removed:
            print(f"ETF filter: {before} → {len(df)} stocks ({removed} removed)")
    # Also drop tickers with common ETF suffixes just in case
    before = len(df)
    df = df[~df["Ticker"].apply(
        lambda t: any(t.endswith(s) for s in ['ETF', 'ETN', 'ETP'])
    )].copy()

    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].reset_index(drop=True)
    return df


def load_industry_rs(url: str) -> dict:
    """Fetch industry RS CSV, return {industry_name: percentile}."""
    if not url:
        print("Warning: INDUSTRIES_CSV_URL not set, rs_12m will be null for all industries")
        return {}
    try:
        df = pd.read_csv(url)
        df["Industry"]   = df["Industry"].astype(str).str.strip()
        df["Percentile"] = pd.to_numeric(df["Percentile"], errors="coerce")
        result = {row["Industry"]: row["Percentile"] for _, row in df.iterrows() if pd.notna(row["Percentile"])}
        print(f"Loaded industry RS for {len(result)} industries from external CSV")
        return result
    except Exception as e:
        print(f"Warning: failed to load industry RS CSV: {e}")
        return {}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def calculate_dist_ma(close: pd.Series, ma_type: str, period: int) -> float | None:
    if len(close) < period:
        return None
    try:
        if ma_type == "EMA":
            ma = close.ewm(span=period, adjust=False).mean().iloc[-1]
        else:
            ma = close.rolling(window=period).mean().iloc[-1]
        current = close.iloc[-1]
        if ma == 0:
            return None
        return round(((current - ma) / ma) * 100, 2)
    except Exception:
        return None


def calculate_ma_value(close: pd.Series, ma_type: str, period: int) -> float | None:
    if len(close) < period:
        return None
    try:
        if ma_type == "EMA":
            ma = close.ewm(span=period, adjust=False).mean().iloc[-1]
        else:
            ma = close.rolling(window=period).mean().iloc[-1]
        return round(float(ma), 4)
    except Exception:
        return None


SLOPE_LOOKBACK_RATIO = 0.15   # 15% of MA period
SLOPE_LOOKBACK_MIN   = 5      # floor: don't go below 5 bars
SLOPE_LOOKBACK_MAX   = 20     # ceiling: don't go above 20 bars

def calculate_slope_ma(close: pd.Series, ma_type: str, period: int) -> float | None:
    """% change of MA over a scaled lookback window (15% of period, clamped 5–20 bars)."""
    lookback = int(round(period * SLOPE_LOOKBACK_RATIO))
    lookback = max(SLOPE_LOOKBACK_MIN, min(SLOPE_LOOKBACK_MAX, lookback))
    if len(close) < period + lookback:
        return None
    try:
        if ma_type == "EMA":
            ma_series = close.ewm(span=period, adjust=False).mean()
        else:
            ma_series = close.rolling(window=period).mean()
        ma_today = ma_series.iloc[-1]
        ma_prev  = ma_series.iloc[-1 - lookback]
        if ma_prev == 0 or pd.isna(ma_prev) or pd.isna(ma_today):
            return None
        return round(((ma_today - ma_prev) / ma_prev) * 100, 2)
    except Exception:
        return None


def calculate_adr_pct(hist: pd.DataFrame, period: int = 20) -> float | None:
    try:
        capped = min(period, len(hist))
        adr = (hist["High"] - hist["Low"]).rolling(window=capped).mean().iloc[-1]
        current = hist["Close"].iloc[-1]
        if current and current > 0:
            return round((adr / current) * 100, 2)
    except Exception:
        pass
    return None


def calculate_rsi14(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI: SMA seed for first `period` bars, then RMA smoothing."""
    if len(close) < period + 1:
        return None
    try:
        delta  = close.diff().iloc[1:].values
        gains  = np.where(delta > 0,  delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    except Exception:
        return None


def calculate_spark_series(close: pd.Series, window: int = SPARK_WINDOW, points: int = SPARK_POINTS) -> list[float] | None:
    """Cumulative % return over the trailing `window` trading days, sampled at
    `points` evenly-spaced steps (~weekly). First point is always 0.0 (the
    rebase anchor); the last point lines up closely with the existing `3m`
    field for the same stock, since both use the same window."""
    if len(close) < window + 1:
        return None
    try:
        idxs = [int(round(x)) for x in np.linspace(-window, -1, points)]
        prices = [close.iloc[i] for i in idxs]
        if any(pd.isna(p) for p in prices):
            return None
        base = prices[0]
        if base == 0:
            return None
        return [round(((p / base) - 1) * 100, 2) for p in prices]
    except Exception:
        return None


def compute_metrics(ticker: str, hist: pd.DataFrame, spy_hist: pd.DataFrame) -> dict | None:
    """Compute all price-derived metrics for a single ticker."""
    try:
        hist = hist.dropna(subset=["Close", "Open", "High", "Low"])
        if len(hist) < 2:
            return None

        close   = hist["Close"]
        current = close.iloc[-1]

        if current < MIN_PRICE:
            return None  # below minimum price threshold
        prev    = close.iloc[-2]

        daily       = (current / prev - 1) * 100
        one_week    = (current / close.iloc[-6]  - 1) * 100 if len(close) >= 6  else None
        one_month   = (current / close.iloc[-22] - 1) * 100 if len(close) >= 22 else None
        three_month = (current / close.iloc[-63] - 1) * 100 if len(close) >= 63 else None
        spark_3m    = calculate_spark_series(close)

        # YTD: first close of current calendar year
        year_rows = hist[hist.index.year == hist.index[-1].year]
        ytd = (current / year_rows["Close"].iloc[0] - 1) * 100 if len(year_rows) >= 1 else None

        # vs SPY relative strength (1m, 3m, 6m, 12m)
        vs_spy_1m = vs_spy_3m = vs_spy_6m = vs_spy_12m = None
        six_month    = (current / close.iloc[-126] - 1) * 100 if len(close) >= 126 else None
        twelve_month = (current / close.iloc[-252] - 1) * 100 if len(close) >= 252 else None
        if spy_hist is not None:
            spy_close = spy_hist["Close"]
            if len(spy_close) >= 22 and len(close) >= 22:
                spy_1m    = (spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100
                vs_spy_1m = round(one_month - spy_1m, 2) if one_month is not None else None
            if len(spy_close) >= 63 and len(close) >= 63:
                spy_3m    = (spy_close.iloc[-1] / spy_close.iloc[-63] - 1) * 100
                vs_spy_3m = round(three_month - spy_3m, 2) if three_month is not None else None
            if len(spy_close) >= 126 and len(close) >= 126:
                spy_6m    = (spy_close.iloc[-1] / spy_close.iloc[-126] - 1) * 100
                vs_spy_6m = round(six_month - spy_6m, 2) if six_month is not None else None
            if len(spy_close) >= 252 and len(close) >= 252:
                spy_12m    = (spy_close.iloc[-1] / spy_close.iloc[-252] - 1) * 100
                vs_spy_12m = round(twelve_month - spy_12m, 2) if twelve_month is not None else None

        # Weighted 3M RS (4×16-day periods, weights 40/20/20/20, vs SPX)
        # Replicates TradingView Pine Script formula
        weighted_rs_score = None
        weighted_rs_pct   = None
        WRS_THRESHOLDS = [
            (165.00, 99), (125.00, 95), (115.00, 90), (108.00, 80),
            (103.00, 70), (100.30, 60), (99.50,  50), (98.50,  40),
            (97.50,  30), (92.00,  20), (82.00,  10), (70.00,   1),
        ]
        try:
            if spy_hist is not None and len(close) >= 65 and len(spy_hist["Close"]) >= 65:
                sc = spy_hist["Close"]
                c, s = close.iloc[-1], sc.iloc[-1]
                stock_w = 0.40*(c/close.iloc[-17]) + 0.20*(c/close.iloc[-33]) + 0.20*(c/close.iloc[-49]) + 0.20*(c/close.iloc[-65])
                ref_w   = 0.40*(s/sc.iloc[-17])    + 0.20*(s/sc.iloc[-33])    + 0.20*(s/sc.iloc[-49])    + 0.20*(s/sc.iloc[-65])
                if abs(ref_w) >= 0.001:
                    raw = (stock_w / ref_w) * 100
                    weighted_rs_score = round(raw, 2)
                    # Map to percentile via thresholds
                    if raw >= WRS_THRESHOLDS[0][0]:
                        weighted_rs_pct = 99
                    elif raw <= WRS_THRESHOLDS[-1][0]:
                        weighted_rs_pct = 1
                    else:
                        for i in range(len(WRS_THRESHOLDS) - 1):
                            hi_v, hi_r = WRS_THRESHOLDS[i]
                            lo_v, lo_r = WRS_THRESHOLDS[i + 1]
                            if raw >= lo_v:
                                t = (raw - lo_v) / (hi_v - lo_v)
                                weighted_rs_pct = round(lo_r + t * (hi_r - lo_r))
                                break
        except Exception:
            pass


        today_high = hist["High"].iloc[-1]
        today_low  = hist["Low"].iloc[-1]
        cr = ((current - today_low) / (today_high - today_low) * 100) \
            if (today_high - today_low) > 0 else None

        # ADR%
        adr_pct = calculate_adr_pct(hist)

        # Relative volume: today's volume vs 50-day average volume
        rel_vol = None
        try:
            today_vol = hist["Volume"].iloc[-1]
            avg_vol   = hist["Volume"].iloc[-51:-1].mean()  # prior 50 days excluding today
            if avg_vol and avg_vol > 0 and not pd.isna(today_vol):
                rel_vol = round(float(today_vol) / float(avg_vol), 2)
        except Exception:
            pass

        # U/D Volume Ratio: sum of up-day volume / sum of down-day volume over 20 and 50 days
        ud_vol_ratio_20 = None
        ud_vol_ratio_50 = None
        try:
            if len(hist) >= 21:
                h20 = hist.iloc[-21:]
                up_vol_20 = h20["Volume"].where(h20["Close"] > h20["Close"].shift(1), 0).iloc[1:].sum()
                dn_vol_20 = h20["Volume"].where(h20["Close"] < h20["Close"].shift(1), 0).iloc[1:].sum()
                if dn_vol_20 and dn_vol_20 > 0:
                    ud_vol_ratio_20 = round(float(up_vol_20) / float(dn_vol_20), 1)
            if len(hist) >= 51:
                h50 = hist.iloc[-51:]
                up_vol_50 = h50["Volume"].where(h50["Close"] > h50["Close"].shift(1), 0).iloc[1:].sum()
                dn_vol_50 = h50["Volume"].where(h50["Close"] < h50["Close"].shift(1), 0).iloc[1:].sum()
                if dn_vol_50 and dn_vol_50 > 0:
                    ud_vol_ratio_50 = round(float(up_vol_50) / float(dn_vol_50), 1)
        except Exception:
            pass

        # Dist/MA for all combos
        dist_ma = {
            ma_type + str(period): calculate_dist_ma(close, ma_type, period)
            for ma_type, period in DIST_MA_COMBOS
        }

        # Slope/MA for all combos (% change of MA over N periods, N = MA period)
        slope_ma = {
            ma_type + str(period): calculate_slope_ma(close, ma_type, period)
            for ma_type, period in DIST_MA_COMBOS
        }

        # Raw MA values (for MA vs MA cross comparisons)
        ma_val = {
            ma_type + str(period): calculate_ma_value(close, ma_type, period)
            for ma_type, period in DIST_MA_COMBOS
        }

        # Yesterday's MA values (for crossover detection)
        close_prev = close.iloc[:-1]
        ma_val_prev = {
            ma_type + str(period): calculate_ma_value(close_prev, ma_type, period)
            for ma_type, period in DIST_MA_COMBOS
        }

        # MA crossovers today: store set of "fast|slow|direction" strings
        # e.g. "SMA5|SMA50|above" means SMA5 crossed above SMA50 today
        ma_crossovers: set[str] = set()
        for ma1, p1 in DIST_MA_COMBOS:
            for ma2, p2 in DIST_MA_COMBOS:
                if ma1 == ma2 and p1 == p2:
                    continue
                k1, k2 = ma1 + str(p1), ma2 + str(p2)
                today1, today2   = ma_val.get(k1),      ma_val.get(k2)
                prev1,  prev2    = ma_val_prev.get(k1), ma_val_prev.get(k2)
                if None in (today1, today2, prev1, prev2):
                    continue
                # Crossed above: was below/equal yesterday, above today
                if prev1 <= prev2 and today1 > today2:
                    ma_crossovers.add(f"{k1}|{k2}|above")
                # Crossed below: was above/equal yesterday, below today
                elif prev1 >= prev2 and today1 < today2:
                    ma_crossovers.add(f"{k1}|{k2}|below")

        # Price MA crossovers today: store set of "MAkey|direction" strings
        # e.g. "SMA50|above" means price crossed above SMA50 today
        price_ma_crossovers: set[str] = set()
        for ma_type, period in DIST_MA_COMBOS:
            key = ma_type + str(period)
            dist_today = dist_ma.get(key)          # today: (price - MA) / MA * 100
            ma_prev    = ma_val_prev.get(key)       # yesterday's MA value
            if dist_today is None or ma_prev is None or ma_prev == 0:
                continue
            dist_prev = (close_prev.iloc[-1] - ma_prev) / ma_prev * 100
            # Crossed above: was at or below MA yesterday, above today
            if dist_prev <= 0 and dist_today > 0:
                price_ma_crossovers.add(f"{key}|above")
            # Crossed below: was at or above MA yesterday, below today
            elif dist_prev >= 0 and dist_today < 0:
                price_ma_crossovers.add(f"{key}|below")

        # Today's OHLC — needed for range metrics and candle patterns below
        _o, _c = hist["Open"].iloc[-1], hist["Close"].iloc[-1]
        _h, _l = hist["High"].iloc[-1], hist["Low"].iloc[-1]

        # Narrow range metrics
        # range_vs_adr: today's H-L range as % of price, divided by ADR% — ratio of today vs average
        # range_rank: how many consecutive prior days today's range is narrower than (up to 20)
        range_vs_adr = None
        range_rank   = None
        try:
            today_range_pct = (_h - _l) / current * 100 if current > 0 else None
            if today_range_pct is not None and adr_pct and adr_pct > 0:
                range_vs_adr = round(today_range_pct / adr_pct * 100, 1)  # e.g. 45 means 45% of ADR
            if len(hist) >= 2:
                today_hl = _h - _l
                max_lookback = min(20, len(hist) - 1)
                rank = 0
                for k in range(1, max_lookback + 1):
                    prev_hl = hist["High"].iloc[-1 - k] - hist["Low"].iloc[-1 - k]
                    if today_hl < prev_hl:
                        rank = k
                    else:
                        break
                range_rank = rank  # e.g. 7 means today is narrowest of last 7 days
        except Exception:
            pass

        # Gap %: today's open vs prior day's high/low (true gap — no range overlap)
        # Gap up:   today_open > prev_high  → gap_pct = (today_open - prev_high) / prev_high * 100  (positive)
        # Gap down: today_open < prev_low   → gap_pct = (today_open - prev_low)  / prev_low  * 100  (negative)
        # No gap: today_open within prev range → gap_pct = 0
        gap_pct = None
        try:
            today_open = hist["Open"].iloc[-1]
            prev_high  = hist["High"].iloc[-2]
            prev_low   = hist["Low"].iloc[-2]
            if prev_high > 0 and prev_low > 0:
                if today_open > prev_high:
                    gap_pct = round((today_open - prev_high) / prev_high * 100, 2)
                elif today_open < prev_low:
                    gap_pct = round((today_open - prev_low) / prev_low * 100, 2)
                else:
                    gap_pct = 0.0
        except Exception:
            pass

        # Inside day: today's high < prev high AND today's low > prev low
        inside_day = bool(
            hist["High"].iloc[-1] < hist["High"].iloc[-2] and
            hist["Low"].iloc[-1]  > hist["Low"].iloc[-2]
        ) if len(hist) >= 2 else False

        # Double inside day: bar[-1] inside bar[-2], AND bar[-2] inside bar[-3]
        double_inside_day = bool(
            hist["High"].iloc[-1] < hist["High"].iloc[-2] and
            hist["Low"].iloc[-1]  > hist["Low"].iloc[-2] and
            hist["High"].iloc[-2] < hist["High"].iloc[-3] and
            hist["Low"].iloc[-2]  > hist["Low"].iloc[-3]
        ) if len(hist) >= 3 else False

        # Bullish outside day: today's high > prev high AND today's low < prev low
        bullish_outside = bool(
            hist["High"].iloc[-1] > hist["High"].iloc[-2] and
            hist["Low"].iloc[-1]  < hist["Low"].iloc[-2]
        ) if len(hist) >= 2 else False

        # Bearish outside day: outside day (engulfs prev range) AND closes below prev close
        bearish_outside = bool(
            hist["High"].iloc[-1] > hist["High"].iloc[-2] and
            hist["Low"].iloc[-1]  < hist["Low"].iloc[-2] and
            hist["Close"].iloc[-1] < hist["Close"].iloc[-2]
        ) if len(hist) >= 2 else False

        # Hammer / Bullish Hammer
        _body        = abs(_c - _o)
        _candle_range = _h - _l
        _lower_shadow = min(_o, _c) - _l
        _upper_shadow = _h - max(_o, _c)
        hammer = bool(
            _candle_range > 0 and
            _body <= 0.3 * _candle_range and
            _lower_shadow >= 2 * _body and
            _upper_shadow <= 0.1 * _candle_range
        )

        _prev_low   = hist["Low"].iloc[-2]
        _prev_close = hist["Close"].iloc[-2]

        # Pocket Pivot: today closes up AND today's volume > max down-volume of prior 10 days
        pocket_pivot = False
        try:
            if len(hist) >= 11:
                today_vol  = hist["Volume"].iloc[-1]
                prior_10   = hist.iloc[-11:-1]
                down_days  = prior_10[prior_10["Close"] < prior_10["Close"].shift(1)]
                max_down_vol = down_days["Volume"].max() if len(down_days) > 0 else 0
                pocket_pivot = bool(
                    _c > _prev_close and
                    today_vol > max_down_vol
                )
        except Exception:
            pass

        # Bullish Reversal Bar: undercuts prev low, closes above prev close
        bullish_reversal_bar = bool(
            _l < _prev_low and
            _c > _prev_close
        ) if len(hist) >= 2 else False

        # Upside Reversal: undercuts prev low, closes in upper half of today's range
        upside_reversal = bool(
            _candle_range > 0 and
            _l < _prev_low and
            _c >= (_l + _candle_range * 0.5)
        ) if len(hist) >= 2 else False

        # Oops Reversal: opens below prev low, closes above prev low
        _prev_low2 = hist["Low"].iloc[-2]
        oops_reversal = bool(
            hist["Open"].iloc[-1] < _prev_low2 and
            _c > _prev_low2
        ) if len(hist) >= 2 else False

        # ── Weekly & Monthly pattern detection ────────────────────────────
        def detect_patterns(h: pd.DataFrame) -> dict:
            """Run all pattern checks on the last 2 bars of a resampled DataFrame."""
            out = {k: False for k in [
                "inside_day", "double_inside_day", "bullish_outside", "bearish_outside", "hammer",
                "bullish_reversal_bar", "upside_reversal",
                "oops_reversal", "pocket_pivot",
            ]}
            if len(h) < 2:
                return out
            h = h.dropna(subset=["Open", "High", "Low", "Close"])
            if len(h) < 2:
                return out
            try:
                o, c = h["Open"].iloc[-1], h["Close"].iloc[-1]
                hi, lo = h["High"].iloc[-1], h["Low"].iloc[-1]
                p_hi, p_lo = h["High"].iloc[-2], h["Low"].iloc[-2]
                p_close = h["Close"].iloc[-2]
                body         = abs(c - o)
                candle_range = hi - lo
                lower_shadow = min(o, c) - lo
                upper_shadow = hi - max(o, c)
                out["inside_day"]     = bool(hi < p_hi and lo > p_lo)
                out["double_inside_day"] = bool(
                    len(h) >= 3 and
                    hi < p_hi and lo > p_lo and
                    p_hi < h["High"].iloc[-3] and p_lo > h["Low"].iloc[-3]
                )
                out["bullish_outside"]= bool(hi > p_hi and lo < p_lo)
                out["bearish_outside"]= bool(hi > p_hi and lo < p_lo and c < p_close)
                out["hammer"]         = bool(
                    candle_range > 0 and body <= 0.3 * candle_range and
                    lower_shadow >= 2 * body and upper_shadow <= 0.1 * candle_range
                )
                out["bullish_reversal_bar"] = bool(lo < p_lo and c > p_close)
                out["upside_reversal"]      = bool(
                    candle_range > 0 and lo < p_lo and c >= (lo + candle_range * 0.5)
                )
                out["oops_reversal"]  = bool(o < p_lo and c > p_lo)
                # Pocket pivot on weekly/monthly: close up AND volume > max down-bar vol of prior 10 bars
                if len(h) >= 11:
                    today_vol2  = h["Volume"].iloc[-1]
                    prior_10_2  = h.iloc[-11:-1]
                    down_bars   = prior_10_2[prior_10_2["Close"] < prior_10_2["Close"].shift(1)]
                    max_dv      = down_bars["Volume"].max() if len(down_bars) > 0 else 0
                    out["pocket_pivot"] = bool(c > p_close and today_vol2 > max_dv)
            except Exception:
                pass
            return out

        # Resample to weekly (week ending Friday) and monthly
        weekly_hist  = hist.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])
        monthly_hist = hist.resample("ME").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Close"])

        weekly_patterns  = detect_patterns(weekly_hist)
        monthly_patterns = detect_patterns(monthly_hist)

        # Weekly and monthly closing range
        def calc_cr(h: pd.DataFrame):
            if len(h) < 1:
                return None
            h = h.dropna(subset=["High", "Low", "Close"])
            if len(h) < 1:
                return None
            try:
                hi = h["High"].iloc[-1]
                lo = h["Low"].iloc[-1]
                cl = h["Close"].iloc[-1]
                return round((cl - lo) / (hi - lo) * 100, 1) if (hi - lo) > 0 else None
            except Exception:
                return None

        cr_w = calc_cr(weekly_hist)
        cr_m = calc_cr(monthly_hist)

        # 52-week high/low — compare today's bar against the prior 252 sessions (excluding today)
        new_52wk_high = False
        new_52wk_low  = False
        pct_from_52wk_low = None
        try:
            prior = hist.iloc[:-1]  # everything except today
            if len(prior) >= 20:    # need enough history to be meaningful
                high_52w = prior["High"].max()
                low_52w  = prior["Low"].min()
                today_high = hist["High"].iloc[-1]
                today_low  = hist["Low"].iloc[-1]
                new_52wk_high = bool(today_high >= high_52w)
                new_52wk_low  = bool(today_low  <= low_52w)
                if low_52w and low_52w > 0:
                    pct_from_52wk_low = round((current / low_52w - 1) * 100, 2)
        except Exception:
            pass

        return {
            "price":      round(float(current), 2),
            "inside_day": inside_day,
            "double_inside_day": double_inside_day,
            "bullish_outside": bullish_outside,
            "bearish_outside": bearish_outside,
            "hammer":              hammer,
            "bullish_reversal_bar": bullish_reversal_bar,
            "upside_reversal":      upside_reversal,
            "oops_reversal":        oops_reversal,
            "pocket_pivot":         pocket_pivot,
            # Weekly patterns
            "inside_day_w":          weekly_patterns["inside_day"],
            "double_inside_day_w":   weekly_patterns["double_inside_day"],
            "bullish_outside_w":     weekly_patterns["bullish_outside"],
            "bearish_outside_w":     weekly_patterns["bearish_outside"],
            "hammer_w":              weekly_patterns["hammer"],
            "bullish_reversal_bar_w":weekly_patterns["bullish_reversal_bar"],
            "upside_reversal_w":     weekly_patterns["upside_reversal"],
            "oops_reversal_w":       weekly_patterns["oops_reversal"],
            "pocket_pivot_w":        weekly_patterns["pocket_pivot"],
            # Monthly patterns
            "inside_day_m":          monthly_patterns["inside_day"],
            "double_inside_day_m":   monthly_patterns["double_inside_day"],
            "bullish_outside_m":     monthly_patterns["bullish_outside"],
            "bearish_outside_m":     monthly_patterns["bearish_outside"],
            "hammer_m":              monthly_patterns["hammer"],
            "bullish_reversal_bar_m":monthly_patterns["bullish_reversal_bar"],
            "upside_reversal_m":     monthly_patterns["upside_reversal"],
            "oops_reversal_m":       monthly_patterns["oops_reversal"],
            "pocket_pivot_m":        monthly_patterns["pocket_pivot"],
            "new_52wk_high":        new_52wk_high,
            "new_52wk_low":         new_52wk_low,
            "PctFrom52WkLow":       pct_from_52wk_low,
            "daily":      round(daily, 2),
            "1w":         round(one_week, 2)    if one_week    is not None else None,
            "1m":         round(one_month, 2)   if one_month   is not None else None,
            "3m":        round(three_month, 2) if three_month is not None else None,
            "spark_3m":  spark_3m,
            "6m":        round(six_month, 2)   if six_month   is not None else None,
            "1y":         round(twelve_month, 2) if twelve_month is not None else None,
            "ytd":       round(ytd, 2)         if ytd         is not None else None,
            "vs_spy":     vs_spy_1m,
            "vs_spy_3m":  vs_spy_3m,
            "vs_spy_6m":  vs_spy_6m,
            "vs_spy_12m": vs_spy_12m,
            "weighted_rs_score": weighted_rs_score,
            "weighted_rs_pct":   weighted_rs_pct,
            "cr":        round(cr, 1)          if cr          is not None else None,
            "cr_w":      cr_w,
            "cr_m":      cr_m,
            "adr_pct":   adr_pct,
            "rel_vol":        rel_vol,
            "ud_vol_ratio_20": ud_vol_ratio_20,
            "ud_vol_ratio_50": ud_vol_ratio_50,
            "dist_ma":   dist_ma,
            "slope_ma":  slope_ma,
            "ma_val":    ma_val,
            "ma_crossovers":       list(ma_crossovers),
            "price_ma_crossovers": list(price_ma_crossovers),
            "gap_pct":             gap_pct,
            "range_vs_adr":        range_vs_adr,
            "range_rank":          range_rank,
            "rsi14":               calculate_rsi14(close),
        }
    except Exception as e:
        print(f"  Metric error [{ticker}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Parallel history fetching
# ---------------------------------------------------------------------------

def fetch_history_batch(tickers: list[str], max_workers: int = 10) -> dict[str, pd.DataFrame]:
    """
    Fetch 14mo daily history for all tickers using yf.download() in 100-ticker chunks.
    Falls back to per-ticker ThreadPoolExecutor for any chunk that fails.
    Returns dict of ticker -> DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    CHUNK = 100

    chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]
    print(f"Downloading history for {len(tickers)} tickers "
          f"in {len(chunks)} chunks of {CHUNK}...")

    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx+1}/{len(chunks)} ({len(chunk)} tickers)...", end=" ", flush=True)
        try:
            raw = yf.download(
                chunk,
                period="14mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if len(chunk) == 1:
                df = raw.dropna(how="all")
                if len(df) >= 2:
                    results[chunk[0]] = df
                ok = 1 if len(df) >= 2 else 0
            else:
                ok = 0
                for ticker in chunk:
                    try:
                        df = raw[ticker].dropna(how="all")
                        if len(df) >= 2:
                            results[ticker] = df
                            ok += 1
                    except Exception:
                        pass
            print(f"ok={ok}/{len(chunk)}")
        except Exception as e:
            print(f"batch failed ({e}), falling back to per-ticker...")
            def _fetch_one(t: str):
                try:
                    df = yf.Ticker(t).history(period="14mo")
                    return t, df if len(df) >= 2 else None
                except Exception:
                    return t, None

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_fetch_one, t): t for t in chunk}
                ok = 0
                for fut in as_completed(futs):
                    t, df = fut.result()
                    if df is not None:
                        results[t] = df
                        ok += 1
            print(f"fallback ok={ok}/{len(chunk)}")

        time.sleep(0.5)  # polite pause between chunks

    print(f"History ready for {len(results)}/{len(tickers)} tickers\n")
    return results


# ---------------------------------------------------------------------------
# JSON sanitize
# ---------------------------------------------------------------------------

def sanitize(obj):
    """Recursively strip NaN/Inf and convert numpy scalars for JSON serialization."""
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",           default="data",                      help="Output directory")
    parser.add_argument("--csv-url",           default=DEFAULT_CSV_URL,             help="URL of RS stock universe CSV")

    parser.add_argument("--workers",           type=int, default=10,                help="Threads for fallback fetches")
    args = parser.parse_args()

    if not args.csv_url:
        raise SystemExit("Error: --csv-url is required (or set CSV_URL env var)")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load fundamentals cache (written weekly by build_fundamentals.yml)
    fundamentals_path = os.path.join(args.out_dir, "fundamentals.json")
    fundamentals_cache: dict = {}
    if os.path.exists(fundamentals_path):
        try:
            with open(fundamentals_path, "r", encoding="utf-8") as f:
                fundamentals_cache = json.load(f).get("fundamentals", {})
            print(f"Loaded fundamentals for {len(fundamentals_cache)} tickers")
        except Exception as e:
            print(f"Warning: could not load fundamentals.json: {e}")
    else:
        print("Warning: fundamentals.json not found, skipping fundamental fields")

    # 1. Load & filter universe
    universe = load_universe(args.csv_url)
    industry_rs_map = load_industry_rs(INDUSTRIES_CSV_URL)

    # Inject supplemental tickers (bypass all filters — manually curated)
    if SUPPLEMENTAL_TICKERS:
        existing = set(universe["Ticker"].tolist())
        new_rows = [t for t in SUPPLEMENTAL_TICKERS if t["Ticker"] not in existing]
        if new_rows:
            supplement_df = pd.DataFrame(new_rows)
            # Ensure all expected columns exist (fill missing with NaN)
            for col in universe.columns:
                if col not in supplement_df.columns:
                    supplement_df[col] = None
            supplement_df = supplement_df[universe.columns]  # align column order
            universe = pd.concat([universe, supplement_df], ignore_index=True)
            print(f"Supplemental tickers added: {[r['Ticker'] for r in new_rows]}")
        else:
            print("Supplemental tickers: all already present in CSV")

    tickers  = universe["Ticker"].tolist()
    print(f"Universe after filters: {len(tickers)} stocks\n")

    # 2. SPX baseline
    print("Fetching ^GSPC history...")
    spy_hist  = yf.Ticker("^GSPC").history(period="14mo").dropna(subset=["Close"])
    spy_last_date = spy_hist.index[-1]  # freshness reference — SPY always has today's bar
    if spy_last_date.tzinfo is not None:
        spy_last_date = spy_last_date.tz_localize(None)  # yf.Ticker().history() is tz-aware; normalize for safe subtraction

    # 3. Batch-fetch price histories
    histories = fetch_history_batch(tickers, max_workers=args.workers)

    # 4. Compute metrics and group by industry
    print("Computing metrics...")
    csv_lookup = universe.set_index("Ticker").to_dict(orient="index")

    by_industry:  dict[str, list]  = defaultdict(list)
    industry_to_sector: dict[str, str] = {}
    industry_spark_series: dict[str, list] = defaultdict(list)  # per-industry list of per-stock spark_3m arrays
    skipped = 0
    skipped_stale = 0
    supplemental_set = {t["Ticker"] for t in SUPPLEMENTAL_TICKERS}

    for ticker in tickers:
        hist = histories.get(ticker)
        if hist is None:
            skipped += 1
            continue

        # Skip tickers with no fresh data — halted, delisted, or otherwise not
        # currently trading. Detected two ways:
        #  1. Last bar is too many calendar days behind SPY's last bar.
        #  2. Last bar exists (feed still returns a stale cached row) but has
        #     zero volume, the classic signature of a halt day.
        last_date = hist.index[-1]
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)  # yf.download() batch results are tz-naive; normalize either way
        lag_days  = (spy_last_date - last_date).days
        if lag_days > STALE_MAX_CALENDAR_DAYS:
            skipped += 1
            skipped_stale += 1
            print(f"  Skipping {ticker}: stale data, {lag_days}d behind SPY "
                  f"(last bar {last_date.date()})")
            continue
        last_volume = hist["Volume"].iloc[-1] if "Volume" in hist.columns else None
        if last_volume is not None and not pd.isna(last_volume) and last_volume == 0:
            skipped += 1
            skipped_stale += 1
            print(f"  Skipping {ticker}: zero volume on last bar "
                  f"({last_date.date()}) — likely halted")
            continue

        metrics = compute_metrics(ticker, hist, spy_hist)
        if metrics is None:
            skipped += 1
            continue

        csv_row  = csv_lookup.get(ticker, {})
        industry = str(csv_row.get("Industry", "Unknown")).strip() or "Unknown"
        sector   = str(csv_row.get("Sector",   "Unknown")).strip() or "Unknown"

        # Apply custom industry override if present
        if ticker in INDUSTRY_OVERRIDES:
            industry = INDUSTRY_OVERRIDES[ticker]["industry"]
            sector   = INDUSTRY_OVERRIDES[ticker]["sector"]

        # Numeric passthrough fields
        passthrough: dict = {}
        for col in CSV_PASSTHROUGH:
            val = csv_row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                passthrough[col] = None
            else:
                try:
                    passthrough[col] = float(val)
                except (ValueError, TypeError):
                    passthrough[col] = str(val)

        row = {
            "ticker":   ticker,
            "sector":   sector,
            "industry": industry,
            **passthrough,
            **metrics,
        }

        # Merge fundamental fields if available
        fund = fundamentals_cache.get(ticker)
        if fund:
            row["eps_this_y_pct"]    = fund.get("eps_this_y_pct")
            row["eps_next_y_pct"]    = fund.get("eps_next_y_pct")
            row["eps_next_5y_pct"]   = fund.get("eps_next_5y_pct")
            row["eps_qoq_pct"]       = fund.get("eps_qoq_pct")
            row["sales_qoq_pct"]     = fund.get("sales_qoq_pct")
            row["profit_margin_pct"] = fund.get("profit_margin_pct")
            row["fwd_pe"]            = fund.get("fwd_pe")
            row["ps_ratio"]          = fund.get("ps_ratio")
            row["peg_ratio"]         = fund.get("peg_ratio")
            row["earnings_date"]     = fund.get("earnings_date")
        else:
            row["eps_this_y_pct"]    = None
            row["eps_next_y_pct"]    = None
            row["eps_next_5y_pct"]   = None
            row["eps_qoq_pct"]       = None
            row["sales_qoq_pct"]     = None
            row["profit_margin_pct"] = None
            row["fwd_pe"]            = None
            row["ps_ratio"]          = None
            row["peg_ratio"]         = None
            row["earnings_date"]     = None

        # For supplemental tickers, compute CSV passthrough fields from price history
        if ticker in supplemental_set and ticker in histories:
            hist = histories[ticker]
            hist_clean = hist.dropna(subset=["Close", "Volume"])
            # AvgVol50 — 50-day average volume (excluding today)
            if row.get("AvgVol50") is None and len(hist_clean) >= 2:
                try:
                    avg_vol = hist_clean["Volume"].iloc[-51:-1].mean()
                    if not pd.isna(avg_vol):
                        row["AvgVol50"] = round(float(avg_vol))
                except Exception:
                    pass
            # PctFrom52WkHigh — % below the 52-week high
            if row.get("PctFrom52WkHigh") is None and len(hist_clean) >= 1:
                try:
                    high_52w = hist_clean["High"].max()
                    current  = hist_clean["Close"].iloc[-1]
                    if high_52w and high_52w > 0:
                        row["PctFrom52WkHigh"] = round((current / high_52w - 1) * 100, 2)
                except Exception:
                    pass
        # Pull the per-stock spark series out before persisting the row — we only
        # ever want the industry-level average in the output, not one array per
        # stock (that would add 13 floats x every stock x every industry for no
        # reason, since this view never renders a per-stock sparkline).
        spark = row.pop("spark_3m", None)
        if spark is not None:
            industry_spark_series[industry].append(spark)

        by_industry[industry].append(row)
        industry_to_sector[industry] = sector

    print(f"Done: {len(tickers) - skipped} computed, {skipped} skipped "
          f"({skipped_stale} stale/halted)\n")

    # Assign Rank to supplemental tickers based on where their Percentile
    # falls relative to the full universe — CSV-sourced ranks are left untouched.
    if supplemental_set:
        # Collect all rows with a Percentile value
        all_rows = [r for rows in by_industry.values() for r in rows]
        ranked_rows = sorted(
            [r for r in all_rows if r.get("Percentile") is not None],
            key=lambda r: float(r["Percentile"]),
            reverse=True,
        )
        # Build a percentile → rank mapping from CSV-sourced tickers only
        # (use their actual assigned ranks to interpolate where supplementals fit)
        csv_ranked = [(float(r["Percentile"]), float(r["Rank"]))
                      for r in ranked_rows
                      if r["ticker"] not in supplemental_set
                      and r.get("Rank") is not None]
        csv_ranked.sort(key=lambda x: x[1])  # sort by rank ascending

        def interpolate_rank(percentile: float) -> int:
            """Find the rank slot where this percentile would fall."""
            if not csv_ranked:
                return 9999
            # Find neighbouring CSV rows by percentile (higher percentile = lower rank number)
            above = [(pct, rnk) for pct, rnk in csv_ranked if pct >= percentile]
            below = [(pct, rnk) for pct, rnk in csv_ranked if pct <  percentile]
            if above and below:
                # Interpolate between the two nearest neighbours
                a_pct, a_rnk = min(above, key=lambda x: x[0])  # closest above
                b_pct, b_rnk = max(below, key=lambda x: x[0])  # closest below
                if a_pct == b_pct:
                    return int(round(a_rnk))
                t = (percentile - b_pct) / (a_pct - b_pct)
                return int(round(b_rnk + t * (a_rnk - b_rnk)))
            elif above:
                return int(round(min(above, key=lambda x: x[0])[1]))
            else:
                return int(round(max(below, key=lambda x: x[0])[1]))

        for row in all_rows:
            if row["ticker"] in supplemental_set and row.get("Percentile") is not None:
                computed_rank = interpolate_rank(float(row["Percentile"]))
                row["Rank"] = computed_rank
                print(f"  Supplemental {row['ticker']}: Percentile={row['Percentile']}, "
                      f"computed Rank={computed_rank}")

    # Sort stocks within each industry by Rank (ascending)
    for rows in by_industry.values():
        rows.sort(key=lambda r: float(r.get("Rank") or 9999))

    # 5. Column ranges per industry (for bar-width scaling in UI)
    def vrange(rows, key, d_min, d_max):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return [min(vals), max(vals)] if vals else [d_min, d_max]

    column_ranges = {
        industry: {
            "daily":     vrange(rows, "daily",     -10,  10),
            "1w":        vrange(rows, "1w",         -20,  20),
            "1m":        vrange(rows, "1m",          -30,  30),
            "3m":        vrange(rows, "3m",          -40,  40),
            "ytd":       vrange(rows, "ytd",         -50,  50),
            "vs_spy":    vrange(rows, "vs_spy",      -20,  20),
            "vs_spy_3m": vrange(rows, "vs_spy_3m",   -30,  30),
        }
        for industry, rows in by_industry.items()
    }

    # 6. Industry-level summary (for top-level industry cards / sorting)
    def avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def avg_dist_ma50(rows):
        vals = [r["dist_ma"]["SMA50"] for r in rows
                if r.get("dist_ma") and r["dist_ma"].get("SMA50") is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def pct_above_dist_ma(rows, key):
        """% of stocks whose dist_ma[key] is positive (price above that MA)."""
        vals = [r["dist_ma"][key] for r in rows
                if r.get("dist_ma") and r["dist_ma"].get(key) is not None]
        return round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1) if vals else None

    def pct_confirmed_uptrend(rows):
        """% of stocks fully stacked in a stage-2 uptrend: Close > 50SMA > 150SMA > 200SMA."""
        total = hits = 0
        for r in rows:
            mv = r.get("ma_val") or {}
            price = r.get("price")
            s50, s150, s200 = mv.get("SMA50"), mv.get("SMA150"), mv.get("SMA200")
            if None in (price, s50, s150, s200):
                continue
            total += 1
            if price > s50 > s150 > s200:
                hits += 1
        return round(hits / total * 100, 1) if total else None

    def median_rs(rows):
        vals = [float(r["Percentile"]) for r in rows if r.get("Percentile") is not None]
        return round(float(np.median(vals)), 1) if vals else None

    def avg_spark_series(series_list):
        """Elementwise average of one industry's per-stock spark_3m arrays."""
        if not series_list:
            return None
        length = len(series_list[0])
        out = []
        for i in range(length):
            vals = [s[i] for s in series_list if i < len(s) and s[i] is not None]
            out.append(round(sum(vals) / len(vals), 2) if vals else None)
        return out

    industry_summary = {
        industry: {
            "sector":          industry_to_sector[industry],
            "stock_count":     len(rows),
            "median_rs_pct":   median_rs(rows),
            "avg_daily":       avg(rows, "daily"),
            "avg_1w":          avg(rows, "1w"),
            "avg_1m":          avg(rows, "1m"),
            "avg_3m":          avg(rows, "3m"),
            "avg_ytd":         avg(rows, "ytd"),
            "avg_vs_spy":      avg(rows, "vs_spy"),
            "avg_vs_spy_3m":   avg(rows, "vs_spy_3m"),
            "avg_6m":           avg(rows, "6m"),
            "avg_1y":           avg(rows, "1y"),
            "avg_vs_spy_6m":   avg(rows, "vs_spy_6m"),
            "avg_vs_spy_12m":  avg(rows, "vs_spy_12m"),
            "avg_dist_ma50":   avg_dist_ma50(rows),
            "pct_confirmed_uptrend": pct_confirmed_uptrend(rows),
            "pct_above_50sma":       pct_above_dist_ma(rows, "SMA50"),
            "pct_above_21ema":       pct_above_dist_ma(rows, "EMA21"),
            "spark_3m":        avg_spark_series(industry_spark_series.get(industry, [])),
        }
        for industry, rows in by_industry.items()
    }

    # 7. Compute industry RS ratings from vs_spy performance
    def ind_rs_thresholds(val, thresholds):
        if val is None:
            return None
        def f_interp(v, hi_v, lo_v, hi_r, lo_r):
            if hi_v == lo_v: return hi_r
            t = (v - lo_v) / (hi_v - lo_v)
            return min(99, max(1, round(lo_r + t * (hi_r - lo_r))))
        if val >= thresholds[0][0]: return 99
        if val <= thresholds[-1][0]: return 1
        for i in range(len(thresholds) - 1):
            hi_v, hi_r = thresholds[i]
            lo_v, lo_r = thresholds[i + 1]
            if val >= lo_v:
                return f_interp(val, hi_v, lo_v, hi_r, lo_r)
        return 1

    RS_THRESHOLDS_1M  = [(31.9,99),(16.9,95),(9.5,90),(4.6,80),(0.9,70),(-0.2,60),(-2.1,50),(-3.5,40),(-4.5,30),(-5.9,20),(-8.6,10),(-20.5,1)]
    RS_THRESHOLDS_3M  = [(65.0,99),(40.0,95),(25.0,90),(13.0,80),(11.0,70),(7.0,60),(4.0,50),(0.0,40),(-3.0,30),(-6.0,20),(-11.0,10),(-20.0,1)]
    RS_THRESHOLDS_6M  = [(80.0,99),(55.0,95),(35.0,90),(20.0,80),(15.0,70),(10.0,60),(5.0,50),(0.0,40),(-5.0,30),(-10.0,20),(-18.0,10),(-30.0,1)]
    for ind, s in industry_summary.items():
        s["rs_1m"]  = ind_rs_thresholds(s.get("avg_vs_spy"),     RS_THRESHOLDS_1M)
        s["rs_3m"]  = ind_rs_thresholds(s.get("avg_vs_spy_3m"),  RS_THRESHOLDS_3M)
        s["rs_6m"]  = ind_rs_thresholds(s.get("avg_vs_spy_6m"),  RS_THRESHOLDS_6M)
        raw_pct = industry_rs_map.get(ind)
        s["rs_12m"] = int(round(raw_pct)) if raw_pct is not None else None

    # 8. Sector → industries index
    sector_to_industries: dict[str, list[str]] = defaultdict(list)
    for industry, sector in industry_to_sector.items():
        sector_to_industries[sector].append(industry)
    for sector in sector_to_industries:
        sector_to_industries[sector].sort()

    # 8. Assemble and write outputs
    snapshot = {
        "built_at":         datetime.utcnow().isoformat() + "Z",
        "by_industry":      dict(by_industry),
        "column_ranges":    column_ranges,
        "industry_summary": industry_summary,
    }
    meta = {
        "sector_to_industries": dict(sector_to_industries),
        "industry_to_sector":   industry_to_sector,
        "all_sectors":          sorted(sector_to_industries.keys()),
        "all_industries":       sorted(by_industry.keys()),
        "default_industry":     sorted(by_industry.keys())[0] if by_industry else "",
        "min_avg_volume":       MIN_AVG_VOLUME,
    }

    # 9. Build industries list with self-computed rank/percentile
    # Blend score: 50% 3m / 25% 1m / 15% 6m / 10% 12m vs SPX
    def blend_score(s):
        v3  = s.get("avg_vs_spy_3m")
        v1  = s.get("avg_vs_spy")
        v6  = s.get("avg_vs_spy_6m")
        v12 = s.get("avg_vs_spy_12m")
        if v3 is None and v1 is None and v6 is None and v12 is None:
            return None
        return round(
            (v3  or 0) * 0.50 +
            (v1  or 0) * 0.25 +
            (v6  or 0) * 0.15 +
            (v12 or 0) * 0.10,
            4
        )

    scored = [
        (industry, blend_score(s))
        for industry, s in industry_summary.items()
    ]
    scored.sort(key=lambda x: x[1] if x[1] is not None else float("-inf"), reverse=True)

    total_industries = len(scored)
    industries_list = []
    for rank_idx, (industry, score) in enumerate(scored, start=1):
        percentile = round((1 - (rank_idx - 1) / max(total_industries - 1, 1)) * 99)
        industries_list.append({
            "rank":        rank_idx,
            "industry":    industry,
            "sector":      industry_to_sector.get(industry, ""),
            "percentile":  percentile,
            "blend_score": score,
        })

    print(f"Computed ranks for {len(industries_list)} industries (50% 3M / 25% 1M / 15% 6M / 10% 12M vs SPX)")

    def write_json(filename, obj):
        path = os.path.join(args.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitize(obj), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"Wrote {path} ({size_kb:.0f} KB)")

    write_json("snapshot.json", snapshot)
    write_json("industries.json", {"built_at": datetime.utcnow().isoformat() + "Z", "industries": industries_list})
    print("\nAll done.")


if __name__ == "__main__":
    main()
