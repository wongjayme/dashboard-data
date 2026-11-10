#!/usr/bin/env python
# rs_ranking.py — Updated version
# New columns in rs_stocks.csv:
#   Price, MarketCap, Float, ShortFloatPct, 52WkHigh, 52WkLow, PctFrom52WkHigh,
#   AvgVol10, AvgVol30, AvgVol50, RevenueGrowth
# Avg volumes calculated from candles (no extra API call).
# Market data from ticker_info.json cache (populated by rs_data.py).

import sys
import pandas as pd
import numpy as np
import json
import os
from datetime import date
from scipy.stats import linregress
import yaml
from rs_data import TD_API, cfg, read_json
from functools import reduce
import datetime

DIR = os.path.dirname(os.path.realpath(__file__))

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
    print(exc)

PRICE_DATA = os.path.join(DIR, "data", "price_history.json")
MIN_PERCENTILE = cfg("MIN_PERCENTILE")
POS_COUNT_TARGET = cfg("POSITIONS_COUNT_TARGET")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)

# ── Column titles ─────────────────────────────────────────────────────────────

TITLE_RANK       = "Rank"
TITLE_TICKER     = "Ticker"
TITLE_TICKERS    = "Tickers"
TITLE_SECTOR     = "Sector"
TITLE_INDUSTRY   = "Industry"
TITLE_UNIVERSE   = "Universe" if not ALL_STOCKS else "Exchange"
TITLE_RS         = "Relative Strength"
TITLE_PERCENTILE = "Percentile"
TITLE_1M         = "1M_RS_Percentile"
TITLE_3M         = "3M_RS_Percentile"
TITLE_6M         = "6M_RS_Percentile"
TITLE_PRICE      = "Price"
TITLE_MKTCAP     = "MarketCap"
TITLE_FLOAT      = "Float"
TITLE_SHORT      = "ShortFloatPct"
TITLE_PCT_52WH   = "PctFrom52WkHigh"
TITLE_AVGVOL10   = "AvgVol10"
TITLE_AVGVOL30   = "AvgVol30"
TITLE_AVGVOL50   = "AvgVol50"
TITLE_REVGROWTH  = "RevenueGrowth"

if not os.path.exists('output'):
    os.makedirs('output')

# ── RS calculation ────────────────────────────────────────────────────────────

def relative_strength(closes: pd.Series, closes_ref: pd.Series):
    rs_stock = strength(closes)
    rs_ref = strength(closes_ref)
    rs = (1 + rs_stock) / (1 + rs_ref) * 100
    rs = int(rs * 100) / 100
    return rs

def strength(closes: pd.Series):
    """Yearly performance, most recent quarter weighted double."""
    try:
        q1 = quarters_perf(closes, 1)
        q2 = quarters_perf(closes, 2)
        q3 = quarters_perf(closes, 3)
        q4 = quarters_perf(closes, 4)
        return 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4
    except:
        return 0

def quarters_perf(closes: pd.Series, n):
    length = min(len(closes), n * int(252 / 4))
    prices = closes.tail(length)
    pct_chg = prices.pct_change().dropna()
    perf_cum = (pct_chg + 1).cumprod() - 1
    return perf_cum.tail(1).item()

# ── Market data helpers ───────────────────────────────────────────────────────

def avg_volume(candles, days):
    """Average daily volume over the last N trading days from candle data."""
    try:
        vols = [c["volume"] for c in candles[-days:] if c.get("volume") is not None]
        return int(sum(vols) / len(vols)) if vols else None
    except Exception:
        return None

def safe_info(ticker, field):
    """Safely retrieve a field from the ticker_info cache."""
    try:
        return TICKER_INFO_DICT[ticker]["info"].get(field)
    except (KeyError, TypeError):
        return None

def pct_from_52wk_high(price, high):
    """% distance from 52-week high. Negative = below high."""
    try:
        if price and high and high > 0:
            return round((price / high - 1) * 100, 2)
        return None
    except Exception:
        return None

# ── TradingView CSV ───────────────────────────────────────────────────────────

def generate_tradingview_csv(percentile_values, first_rs_values):
    lines = []
    trading_days = 0
    yesterday = datetime.date.today() - datetime.timedelta(days=1)

    for percentile in sorted(percentile_values):
        rs_value = first_rs_values[percentile]
        for _ in range(5):
            trading_date = yesterday - datetime.timedelta(days=trading_days)
            date_str = trading_date.strftime("%Y%m%dT")
            lines.append(f"{date_str},0,1000,0,{rs_value},0\n")
            trading_days += 1

    return ''.join(reversed(lines))

# ── Rankings ──────────────────────────────────────────────────────────────────

def rankings():
    """Returns DataFrames with percentile rankings + enriched market data."""
    price_data = read_json(PRICE_DATA)
    rows = []
    ranks = []
    industries = {}
    ind_ranks = []
    stock_rs = {}
    ref = price_data[REFERENCE_TICKER]

    for ticker in price_data:
        if not cfg("SP500") and price_data[ticker].get("universe") == "S&P 500":
            continue
        if not cfg("SP400") and price_data[ticker].get("universe") == "S&P 400":
            continue
        if not cfg("SP600") and price_data[ticker].get("universe") == "S&P 600":
            continue
        if not cfg("NQ100") and price_data[ticker].get("universe") == "Nasdaq 100":
            continue
        try:
            candles    = price_data[ticker]["candles"]
            closes     = [c["close"] for c in candles]
            closes_ref = [c["close"] for c in ref["candles"]]

            industry = (
                TICKER_INFO_DICT[ticker]["info"]["industry"]
                if price_data[ticker].get("industry") == "unknown"
                else price_data[ticker].get("industry", "unknown")
            )
            sector = (
                TICKER_INFO_DICT[ticker]["info"]["sector"]
                if price_data[ticker].get("sector") == "unknown"
                else price_data[ticker].get("sector", "unknown")
            )

            # Skip tickers with no sector AND no industry — likely ETFs or
            # structured products that slipped through the NASDAQ ETF filter
            no_data = ("unknown", "n/a", "Unknown", "N/A", None, "")
            if sector in no_data and industry in no_data:
                continue

            if len(closes) >= 6 * 20:
                cs  = pd.Series(closes)
                csr = pd.Series(closes_ref)
                rs   = relative_strength(cs, csr)
                m    = 20
                rs1m = relative_strength(cs.head(-1 * m), csr.head(-1 * m))
                rs3m = relative_strength(cs.head(-3 * m), csr.head(-3 * m))
                rs6m = relative_strength(cs.head(-6 * m), csr.head(-6 * m))

                if rs < 590:
                    # Price from last candle (most reliable — from our own data)
                    price = round(closes[-1], 2) if closes[-1] else None

                    # Avg volumes from candles — no extra API call
                    av10 = avg_volume(candles, 10)
                    av30 = avg_volume(candles, 30)
                    av50 = avg_volume(candles, 50)
                    rev_growth = safe_info(ticker, "revenueGrowth")

                    # Market data from ticker_info cache
                    mktcap = safe_info(ticker, "marketCap")
                    flt    = safe_info(ticker, "floatShares")
                    short  = safe_info(ticker, "shortPercentOfFloat")
                    wk52h  = safe_info(ticker, "fiftyTwoWeekHigh")  # kept for pct calc only
                    pct52h = pct_from_52wk_high(price, wk52h)

                    ranks.append(len(ranks) + 1)
                    rows.append((
                        0,                                        # Rank placeholder
                        ticker, sector, industry,
                        price_data[ticker].get("universe", ""),
                        rs, 100,                                  # Percentile placeholder
                        rs1m, rs3m, rs6m,
                        price, mktcap, flt, short,
                        pct52h,
                        av10, av30, av50, rev_growth
                    ))
                    stock_rs[ticker] = rs

                    # Industries aggregation
                    if industry not in industries:
                        industries[industry] = {
                            "info": (0, industry, sector, 0, 99, 1, 3, 6),
                            TITLE_RS: [], TITLE_1M: [], TITLE_3M: [],
                            TITLE_6M: [], TITLE_TICKERS: []
                        }
                        ind_ranks.append(len(ind_ranks) + 1)
                    industries[industry][TITLE_RS].append(rs)
                    industries[industry][TITLE_1M].append(rs1m)
                    industries[industry][TITLE_3M].append(rs3m)
                    industries[industry][TITLE_6M].append(rs6m)
                    industries[industry][TITLE_TICKERS].append(ticker)

        except KeyError:
            print(f'Ticker {ticker} has corrupted data.')

    dfs = []

    # ── Stocks DataFrame ──────────────────────────────────────────────────────
    cols = [
        TITLE_RANK, TITLE_TICKER, TITLE_SECTOR, TITLE_INDUSTRY, TITLE_UNIVERSE,
        TITLE_RS, TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M,
        TITLE_PRICE, TITLE_MKTCAP, TITLE_FLOAT, TITLE_SHORT,
        TITLE_PCT_52WH,
        TITLE_AVGVOL10, TITLE_AVGVOL30, TITLE_AVGVOL50, TITLE_REVGROWTH
    ]
    df = pd.DataFrame(rows, columns=cols)

    df[TITLE_PERCENTILE] = pd.qcut(df[TITLE_RS],  100, labels=False, duplicates="drop")
    df[TITLE_1M]         = pd.qcut(df[TITLE_1M],  100, labels=False, duplicates="drop")
    df[TITLE_3M]         = pd.qcut(df[TITLE_3M],  100, labels=False, duplicates="drop")
    df[TITLE_6M]         = pd.qcut(df[TITLE_6M],  100, labels=False, duplicates="drop")

    df = df.sort_values([TITLE_RS], ascending=False)
    df[TITLE_RANK] = list(range(1, len(df) + 1))

    out_tickers_count = int((df[TITLE_PERCENTILE] >= MIN_PERCENTILE).sum())
    df = df.head(out_tickers_count)

    # ── TradingView RSRATING.csv ──────────────────────────────────────────────
    percentile_values = [98, 89, 69, 49, 29, 9, 1]
    first_rs_values = {}

    for percentile in percentile_values:
        matching = df[df[TITLE_PERCENTILE] == percentile]
        if matching.empty:
            available = df[TITLE_PERCENTILE].dropna().unique()
            if len(available) == 0:
                continue
            nearest = min(available, key=lambda x: abs(x - percentile))
            matching = df[df[TITLE_PERCENTILE] == nearest]
        if not matching.empty:
            first_rs_values[percentile] = matching.iloc[0][TITLE_RS]

    if len(first_rs_values) == len(percentile_values):
        tv_csv = generate_tradingview_csv(percentile_values, first_rs_values)
        with open(os.path.join(DIR, "output", "RSRATING.csv"), "w") as f:
            f.write(tv_csv)
        print("✓ RSRATING.csv generated for TradingView.")
    else:
        print("⚠ Could not generate RSRATING.csv — not enough percentile data points.")

    # ── Split into two CSVs to stay under GitHub 500KB display limit ─────────
    # rs_stocks_1.csv → Percentile 50-99 (strongest stocks, most useful)
    # rs_stocks_2.csv → Percentile 0-49  (rest)
    df_top = df[df[TITLE_PERCENTILE] >= 50].copy()
    df_bot = df[df[TITLE_PERCENTILE] <  50].copy()

    df_top.to_csv(os.path.join(DIR, "output", "rs_stocks_1.csv"), index=False)
    df_bot.to_csv(os.path.join(DIR, "output", "rs_stocks_2.csv"), index=False)

    # Full file kept for programmatic use (not rendered by GitHub but downloadable)
    df.to_csv(os.path.join(DIR, "output", "rs_stocks.csv"), index=False)

    print(f"✓ rs_stocks_1.csv: {len(df_top)} tickers (Percentile 50-99)")
    print(f"✓ rs_stocks_2.csv: {len(df_bot)} tickers (Percentile 0-49)")
    dfs.append(df)

    # ── Industries DataFrame ──────────────────────────────────────────────────
    def getDfView(entry):
        return entry["info"]

    def rs_sum(a, b):
        return a + b

    def getRsAverage(ind_dict, industry, column):
        vals = ind_dict[industry][column]
        return int((reduce(rs_sum, vals) / len(vals)) * 100) / 100

    def rs_for_stock(t):
        return stock_rs.get(t, 0)

    def getTickers(ind_dict, industry):
        return ",".join(sorted(ind_dict[industry][TITLE_TICKERS], key=rs_for_stock, reverse=True))

    filtered = list(filter(lambda i: len(i[TITLE_TICKERS]) > 1, list(industries.values())))
    df_ind = pd.DataFrame(
        map(getDfView, filtered),
        columns=[TITLE_RANK, TITLE_INDUSTRY, TITLE_SECTOR, TITLE_RS,
                 TITLE_PERCENTILE, TITLE_1M, TITLE_3M, TITLE_6M]
    )

    df_ind[TITLE_RS]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_RS),  axis=1)
    df_ind[TITLE_1M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_1M),  axis=1)
    df_ind[TITLE_3M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_3M),  axis=1)
    df_ind[TITLE_6M]  = df_ind.apply(lambda r: getRsAverage(industries, r[TITLE_INDUSTRY], TITLE_6M),  axis=1)

    df_ind[TITLE_PERCENTILE] = pd.qcut(df_ind[TITLE_RS], 100, labels=False, duplicates="drop")
    df_ind[TITLE_1M]         = pd.qcut(df_ind[TITLE_1M], 100, labels=False, duplicates="drop")
    df_ind[TITLE_3M]         = pd.qcut(df_ind[TITLE_3M], 100, labels=False, duplicates="drop")
    df_ind[TITLE_6M]         = pd.qcut(df_ind[TITLE_6M], 100, labels=False, duplicates="drop")

    df_ind[TITLE_TICKERS] = df_ind.apply(lambda r: getTickers(industries, r[TITLE_INDUSTRY]), axis=1)
    df_ind = df_ind.sort_values([TITLE_RS], ascending=False)
    df_ind[TITLE_RANK] = list(range(1, len(df_ind) + 1))

    df_ind.to_csv(os.path.join(DIR, "output", 'rs_industries.csv'), index=False)
    dfs.append(df_ind)

    return dfs


def main(skipEnter=False):
    ranks = rankings()
    print(ranks[0])
    print("***\nYour 'rs_stocks.csv' is in the output folder.\n***")
    if not skipEnter and cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")

if __name__ == "__main__":
    main()
