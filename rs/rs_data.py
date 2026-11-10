#!/usr/bin/env python
# rs_data.py — Fixed version
# Changes vs original:
#   1. Python 3.10+ compatible (removed FTP, fixed imports)
#   2. NASDAQ ticker list via HTTPS instead of FTP
#   3. Batch yfinance downloads (100 tickers at a time) → ~25 min instead of hours
#   4. Removed broken session/user-agent hack (curl_cffi handles this transparently)
#   5. pandas 2.x compatible (MultiIndex column handling)

import requests
import json
import time
import datetime as dt
import os
import yaml
import yfinance as yf
import pandas as pd
import dateutil.relativedelta
import numpy as np
import re
from io import StringIO
from time import sleep

from datetime import date
from datetime import datetime

DIR = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(DIR, 'data')):
    os.makedirs(os.path.join(DIR, 'data'))
if not os.path.exists(os.path.join(DIR, 'tmp')):
    os.makedirs(os.path.join(DIR, 'tmp'))

# ── Config ──────────────────────────────────────────────────────────────────

try:
    with open(os.path.join(DIR, 'config_private.yaml'), 'r') as stream:
        private_config = yaml.safe_load(stream)
except FileNotFoundError:
    private_config = None
except yaml.YAMLError as exc:
    print(exc)

try:
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
    print(exc)

def cfg(key):
    try:
        return private_config[key]
    except:
        try:
            return config[key]
        except:
            return None

def read_json(json_file):
    with open(json_file, "r", encoding="utf-8") as fp:
        return json.load(fp)

# ── Constants ────────────────────────────────────────────────────────────────

API_KEY = cfg("API_KEY")
TD_API = "https://api.tdameritrade.com/v1/marketdata/%s/pricehistory"
PRICE_DATA_FILE = os.path.join(DIR, "data", "price_history.json")
REFERENCE_TICKER = cfg("REFERENCE_TICKER")
DATA_SOURCE = cfg("DATA_SOURCE")
ALL_STOCKS = cfg("USE_ALL_LISTED_STOCKS")
TICKER_INFO_FILE = os.path.join(DIR, "data_persist", "ticker_info.json")
TICKER_INFO_DICT = read_json(TICKER_INFO_FILE)
REF_TICKER = {
    "ticker": REFERENCE_TICKER,
    "sector": "--- Reference ---",
    "industry": "--- Reference ---",
    "universe": "--- Reference ---"
}

UNKNOWN = "unknown"

# Exchange code mapping: yfinance codes → human-readable exchange names
YFINANCE_EXCHANGE_MAP = {
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NCM": "NASDAQ",
    "NGM": "NASDAQ",
    "ASE": "AMEX",
    "ARC": "NYSE ARCA",
    "BTS": "BATS",
}

# Batch size for yfinance bulk downloads — 100 is a sweet spot (speed vs reliability)
BATCH_SIZE = 100

# ── Ticker list retrieval ────────────────────────────────────────────────────

def _find_ticker_sector_cols(df, ticker_candidates, sector_candidates, industry_candidates):
    """
    Detect column names dynamically from a list of possible names.
    Wikipedia occasionally renames columns — this makes the code resilient to that.
    """
    cols = [str(c).strip() for c in df.columns]

    def find(candidates):
        for c in candidates:
            for col in cols:
                if c.lower() in col.lower():
                    return col
        return None

    ticker_col    = find(ticker_candidates)
    sector_col    = find(sector_candidates)
    industry_col  = find(industry_candidates)
    return ticker_col, sector_col, industry_col


def get_sp500_tickers(universe="S&P 500"):
    """
    Parse S&P 500 list from Wikipedia using pd.read_html() — robust against HTML structure changes.
    Columns on the page: Symbol | Security | GICS Sector | GICS Sub-Industry | ...
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, attrs={"class": "wikitable"}, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        # Fallback: fetch manually then parse
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    df = tables[0]  # First table = current components
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Symbol", "Ticker", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  S&P 500: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_nq100_tickers(universe="Nasdaq 100"):
    """
    Parse Nasdaq-100 list from Wikipedia.
    Columns: Company | Ticker | GICS Sector | GICS Sub-Industry | ...
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    # The NQ100 component table is usually the largest one on the page
    df = max(tables, key=lambda t: len(t))
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Ticker", "Symbol", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  Nasdaq 100: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_sp_midsmall_tickers(url, universe):
    """
    Generic parser for S&P 400 and S&P 600 Wikipedia pages.
    Columns vary but always contain ticker, sector, and industry info.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rs-rating-bot/1.0)"}
    try:
        tables = pd.read_html(url, header=0, storage_options={"User-Agent": headers["User-Agent"]})
    except Exception:
        resp = requests.get(url, headers=headers, timeout=30)
        tables = pd.read_html(StringIO(resp.text), header=0)

    df = max(tables, key=lambda t: len(t))
    ticker_col, sector_col, industry_col = _find_ticker_sector_cols(
        df,
        ticker_candidates=["Ticker", "Symbol", "Tick"],
        sector_candidates=["GICS Sector", "Sector"],
        industry_candidates=["GICS Sub-Industry", "Sub-Industry", "Industry"]
    )

    secs = {}
    for _, row in df.iterrows():
        ticker = str(row[ticker_col]).strip().replace(".", "-") if ticker_col else None
        if not ticker or not re.match(r'^[A-Z\-]+$', ticker):
            continue
        secs[ticker] = {
            "ticker": ticker,
            "sector":   str(row[sector_col]).strip()   if sector_col   else UNKNOWN,
            "industry": str(row[industry_col]).strip() if industry_col else UNKNOWN,
            "universe": universe
        }
    print(f"  {universe}: {len(secs)} tickers loaded from Wikipedia.")
    return secs


def get_resolved_securities():
    tickers = {REFERENCE_TICKER: REF_TICKER}
    if ALL_STOCKS:
        return get_tickers_from_nasdaq(tickers)
    else:
        return get_tickers_from_wikipedia(tickers)

def get_tickers_from_wikipedia(tickers):
    """
    Load index components from Wikipedia using pd.read_html() with dynamic column detection.
    Much more robust than BeautifulSoup + positional offsets.
    """
    if cfg("NQ100"):
        try:
            tickers.update(get_nq100_tickers())
        except Exception as e:
            print(f"⚠ Could not load Nasdaq-100 from Wikipedia: {e}")
    if cfg("SP500"):
        try:
            tickers.update(get_sp500_tickers())
        except Exception as e:
            print(f"⚠ Could not load S&P 500 from Wikipedia: {e}")
    if cfg("SP400"):
        try:
            tickers.update(get_sp_midsmall_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400"
            ))
        except Exception as e:
            print(f"⚠ Could not load S&P 400 from Wikipedia: {e}")
    if cfg("SP600"):
        try:
            tickers.update(get_sp_midsmall_tickers(
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600"
            ))
        except Exception as e:
            print(f"⚠ Could not load S&P 600 from Wikipedia: {e}")
    return tickers

def exchange_from_symbol(symbol):
    """Map NASDAQ trader file single-letter exchange codes to display names."""
    mapping = {"Q": "NASDAQ", "A": "NYSE ARCA", "N": "NYSE", "P": "NYSE ARCA", "Z": "BATS", "V": "IEXG"}
    return mapping.get(symbol, "n/a")

def exchange_from_yfinance(code):
    """Map yfinance exchange codes (e.g. NYQ, NMS) to display names."""
    return YFINANCE_EXCHANGE_MAP.get(code, code if code else "n/a")

def get_tickers_from_nasdaq(tickers):
    """
    Download the NASDAQ ticker list via HTTPS (replaces old FTP method which was fragile).
    URL: https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt
    """
    print("Downloading NASDAQ ticker list via HTTPS...")
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        lines = r.text.splitlines()
    except Exception as e:
        print(f"HTTPS download failed ({e}), trying fallback URL...")
        # Fallback to alternate URL
        url2 = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqtraded.txt"
        r = requests.get(url2, timeout=60)
        lines = r.text.splitlines()

    # Columns: Nasdaq Traded | Symbol | Security Name | Listing Exchange | Market Category |
    #          ETF | Round Lot Size | Test Issue | Financial Status | ...
    ticker_col = 1
    etf_col = 5
    exchange_col = 3
    test_col = 7

    count_added = 0
    for entry in lines[1:]:  # skip header
        values = entry.split('|')
        if len(values) <= max(ticker_col, etf_col, exchange_col, test_col):
            continue
        ticker = values[ticker_col].strip()
        if (re.match(r'^[A-Z]+$', ticker)
                and values[etf_col].strip() == "N"
                and values[test_col].strip() == "N"):
            sec = {
                "ticker": ticker,
                "sector": UNKNOWN,
                "industry": UNKNOWN,
                "universe": exchange_from_symbol(values[exchange_col].strip())
            }
            tickers[ticker] = sec
            count_added += 1

    print(f"NASDAQ list: {count_added} tickers loaded.")
    return tickers

SECURITIES = get_resolved_securities().values()

# ── File helpers ─────────────────────────────────────────────────────────────

def write_to_file(dict_data, file):
    with open(file, "w", encoding='utf8') as fp:
        json.dump(dict_data, fp, ensure_ascii=False)

def write_price_history_file(tickers_dict):
    write_to_file(tickers_dict, PRICE_DATA_FILE)

def write_ticker_info_file(info_dict):
    write_to_file(info_dict, TICKER_INFO_FILE)

def enrich_ticker_data(ticker_response, security):
    ticker_response["sector"] = security["sector"]
    ticker_response["industry"] = security["industry"]
    ticker_response["universe"] = security["universe"]

# ── Progress helpers ──────────────────────────────────────────────────────────

def print_data_progress(label, idx, total, elapsed_s, remaining_s):
    dt_ref = datetime.fromtimestamp(0)
    dt_e = datetime.fromtimestamp(elapsed_s)
    elapsed = dateutil.relativedelta.relativedelta(dt_e, dt_ref)
    if remaining_s and not np.isnan(remaining_s):
        dt_r = datetime.fromtimestamp(remaining_s)
        remaining = dateutil.relativedelta.relativedelta(dt_r, dt_ref)
        remaining_string = f'{remaining.hours}h {remaining.minutes}m {remaining.seconds}s'
    else:
        remaining_string = "?"
    print(f'[{idx}/{total}] {label} — Elapsed: {elapsed.hours}h {elapsed.minutes}m {elapsed.seconds}s | Remaining: {remaining_string}')

def get_remaining_seconds(all_load_times, idx, total):
    if idx == 0:
        return float('nan')
    load_time_ma = pd.Series(all_load_times).rolling(min(idx + 1, 25)).mean().tail(1).item()
    return (total - idx) * load_time_ma

def escape_ticker(ticker):
    return ticker.replace(".", "-")

# ── Industry/sector info ──────────────────────────────────────────────────────

def get_info_from_dict(d, key):
    return d[key] if key in d else "n/a"

def _safe_float(d, key):
    """Return float value from dict or None if missing/invalid."""
    try:
        v = d.get(key) if hasattr(d, 'get') else getattr(d, key, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None

def load_ticker_info(ticker, info_dict):
    """
    Fetch and cache ticker metadata.
    Uses fast_info (yfinance 1.x, snake_case) for price/market data — much more reliable.
    Falls back to .info for sector/industry/float/short which aren't in fast_info.
    """
    escaped = escape_ticker(ticker)
    result = {
        "industry": "n/a", "sector": "n/a",
        "marketCap": None, "floatShares": None,
        "shortPercentOfFloat": None,
        "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
        "revenueGrowth": None,
    }

    try:
        t = yf.Ticker(escaped)

        # ── fast_info: reliable in yfinance 1.x, camelCase keys ─────────────
        try:
            fi = t.fast_info
            result["marketCap"]        = _safe_float(fi, "marketCap")
            result["fiftyTwoWeekHigh"] = _safe_float(fi, "yearHigh")
            result["fiftyTwoWeekLow"]  = _safe_float(fi, "yearLow")
            result["floatShares"]      = _safe_float(fi, "shares")
        except Exception:
            pass

        # ── .info: uniquement pour sector/industry/short qui ne sont pas dans fast_info ──
        try:
            info_obj = t.info
            result["industry"]            = get_info_from_dict(info_obj, "industry")
            result["sector"]              = get_info_from_dict(info_obj, "sector")
            result["shortPercentOfFloat"] = (
                _safe_float(info_obj, "shortPercentOfFloat") or
                _safe_float(info_obj, "shortRatio")
            )
            result["revenueGrowth"] = _safe_float(info_obj, "revenueGrowth")
            # marketCap fallback si fast_info n'a rien retourné
            if result["marketCap"] is None:
                result["marketCap"] = _safe_float(info_obj, "marketCap")
            # Map yfinance exchange code to readable name
            result["exchange"] = exchange_from_yfinance(info_obj.get("exchange", ""))
        except Exception:
            pass

    except Exception:
        pass

    info_dict[ticker] = {"info": result}

def needs_refresh(ticker, info_dict):
    """
    True if cached entry is missing the new market data fields.
    Triggers a one-time re-fetch on first run after this update.
    """
    try:
        info = info_dict[ticker]["info"]
        return "marketCap" not in info or "revenueGrowth" not in info or "exchange" not in info
    except (KeyError, TypeError):
        return True

# ── Core: batch price download from Yahoo ─────────────────────────────────────

def candles_from_df_row(row):
    """Convert a single-row dict from the batch DataFrame into a candle dict."""
    return {
        "open":     float(row["Open"])   if not pd.isna(row["Open"])   else None,
        "close":    float(row["Close"])  if not pd.isna(row["Close"])  else None,
        "low":      float(row["Low"])    if not pd.isna(row["Low"])    else None,
        "high":     float(row["High"])   if not pd.isna(row["High"])   else None,
        "volume":   float(row["Volume"]) if not pd.isna(row["Volume"]) else None,
        "datetime": int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else 0,
    }

def parse_batch_download(df, batch_tickers):
    """
    Parse the result of yf.download() for multiple tickers.
    yfinance returns a MultiIndex DataFrame: columns = (Field, Ticker).
    Returns a dict: {ticker: [candle, ...]}
    """
    result = {}

    if df is None or df.empty:
        return result

    # Single ticker: flat columns (no MultiIndex)
    if not isinstance(df.columns, pd.MultiIndex):
        ticker = batch_tickers[0]
        df_t = df.copy()
        if df_t.empty:
            return result
        candles = []
        for ts, row in df_t.iterrows():
            if pd.isna(row.get("Close")):
                continue
            candle = {
                "open":     float(row["Open"])   if not pd.isna(row["Open"])   else None,
                "close":    float(row["Close"]),
                "low":      float(row["Low"])    if not pd.isna(row["Low"])    else None,
                "high":     float(row["High"])   if not pd.isna(row["High"])   else None,
                "volume":   float(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "datetime": int(ts.timestamp()),
            }
            candles.append(candle)
        if candles:
            result[ticker] = candles
        return result

    # Multiple tickers: MultiIndex columns (Field, Ticker)
    # Swap levels so we can access by ticker first
    df_swapped = df.swaplevel(axis=1)

    for ticker in batch_tickers:
        # yfinance may have altered the ticker name (e.g. "BRK.B" -> "BRK-B")
        yf_ticker = escape_ticker(ticker)
        try:
            df_t = df_swapped[yf_ticker] if yf_ticker in df_swapped.columns.get_level_values(0) else df_swapped.get(yf_ticker)
            if df_t is None or df_t.empty:
                continue
            # drop rows where Close is NaN
            df_t = df_t.dropna(subset=["Close"])
            if df_t.empty:
                continue
            candles = []
            for ts, row in df_t.iterrows():
                candle = {
                    "open":     float(row["Open"])   if "Open"   in row and not pd.isna(row["Open"])   else None,
                    "close":    float(row["Close"]),
                    "low":      float(row["Low"])    if "Low"    in row and not pd.isna(row["Low"])    else None,
                    "high":     float(row["High"])   if "High"   in row and not pd.isna(row["High"])   else None,
                    "volume":   float(row["Volume"]) if "Volume" in row and not pd.isna(row["Volume"]) else None,
                    "datetime": int(ts.timestamp()),
                }
                candles.append(candle)
            if candles:
                result[ticker] = candles
        except (KeyError, TypeError):
            # ticker not in this batch result — silently skip
            pass

    return result

def load_prices_from_yahoo(securities, info={}):
    """
    Main download function.
    Downloads in batches of BATCH_SIZE tickers for speed (target: ~25 min for 6000+ tickers).
    curl_cffi (installed via requirements.txt) is used transparently by yfinance to bypass
    Yahoo's anti-bot measures — no manual session/user-agent tricks needed.
    """
    print("*** Loading Stocks from Yahoo Finance (batch mode) ***")
    today = date.today()
    start_date = today - dt.timedelta(days=1 * 365 + 183)  # 18 months of data
    end_date = today

    securities_list = list(securities)
    tickers_dict = {}
    failed_tickers = []

    # Separate the reference ticker — download it first, alone
    ref_ticker = REFERENCE_TICKER
    ref_sec = next((s for s in securities_list if s["ticker"] == ref_ticker), None)
    non_ref = [s for s in securities_list if s["ticker"] != ref_ticker]

    # All securities to process (reference first so it's always available)
    ordered = ([ref_sec] if ref_sec else []) + non_ref

    # Build list of (batch_of_securities) chunks
    batches = []
    batch_securities = []
    for sec in ordered:
        batch_securities.append(sec)
        if len(batch_securities) >= BATCH_SIZE:
            batches.append(batch_securities)
            batch_securities = []
    if batch_securities:
        batches.append(batch_securities)

    total_batches = len(batches)
    global_start = time.time()
    batch_times = []

    for batch_idx, batch in enumerate(batches):
        batch_start = time.time()
        batch_tickers = [escape_ticker(s["ticker"]) for s in batch]
        original_tickers = [s["ticker"] for s in batch]

        print(f"\n── Batch {batch_idx + 1}/{total_batches}: {len(batch)} tickers ──")

        try:
            # yfinance batch download
            # auto_adjust=True → split/dividend adjusted prices (default since 0.2.x)
            # group_by='ticker' → MultiIndex (Ticker, Field) — we swap it in parse_batch_download
            df = yf.download(
                tickers=batch_tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
                ignore_tz=True,
            )
        except Exception as e:
            print(f"Batch {batch_idx + 1} failed: {e}")
            failed_tickers.extend(original_tickers)
            continue

        # Parse the downloaded DataFrame into per-ticker candle lists
        candles_by_ticker = parse_batch_download(df, batch_tickers)

        # Map escaped tickers back to original, enrich with security metadata
        escaped_to_original = {escape_ticker(t): t for t in original_tickers}

        for esc_ticker, candles in candles_by_ticker.items():
            original = escaped_to_original.get(esc_ticker, esc_ticker)
            sec = next((s for s in batch if s["ticker"] == original), None)
            if sec is None:
                continue

            ticker_data = {"candles": candles}
            enrich_ticker_data(ticker_data, sec)

            # Industry / sector / market data info (cached in ticker_info.json)
            # Re-fetch if missing OR if new fields (marketCap etc.) not yet in cache
            if original not in TICKER_INFO_DICT or needs_refresh(original, TICKER_INFO_DICT):
                try:
                    load_ticker_info(original, TICKER_INFO_DICT)
                except Exception as e:
                    print(f"  Could not load info for {original}: {e}")

            try:
                ticker_data["industry"] = TICKER_INFO_DICT[original]["info"]["industry"]
                ticker_data["sector"]   = TICKER_INFO_DICT[original]["info"]["sector"]
            except (KeyError, TypeError):
                ticker_data["industry"] = "Unknown"

            # Fallback: if universe wasn't resolved from the NASDAQ listing file,
            # use the exchange name derived from yfinance (fixes NYQ/NMS/ASE etc.)
            try:
                yf_exchange = TICKER_INFO_DICT[original]["info"].get("exchange")
                if yf_exchange and ticker_data.get("universe") in (None, "n/a", UNKNOWN):
                    ticker_data["universe"] = yf_exchange
            except (KeyError, TypeError):
                pass

            tickers_dict[original] = ticker_data

        # Track tickers with no data
        for t in original_tickers:
            if t not in tickers_dict:
                failed_tickers.append(t)

        # Timing
        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)
        total_elapsed = time.time() - global_start
        processed_batches = batch_idx + 1
        avg_batch_time = total_elapsed / processed_batches
        remaining_batches = total_batches - processed_batches
        remaining_s = avg_batch_time * remaining_batches

        remaining_str = f"{int(remaining_s // 60)}m {int(remaining_s % 60)}s" if remaining_s else "?"
        tickers_ok = sum(1 for t in original_tickers if t in tickers_dict)
        print(f"  {tickers_ok}/{len(batch)} tickers OK | Batch time: {batch_elapsed:.1f}s | Remaining: {remaining_str}")

        # Periodic intermediate save (every 10 batches = ~1000 tickers)
        if (batch_idx + 1) % 10 == 0:
            print(f"  → Saving intermediate results ({len(tickers_dict)} tickers so far)...")
            write_price_history_file(tickers_dict)
            write_ticker_info_file(TICKER_INFO_DICT)

    # Final save
    write_price_history_file(tickers_dict)
    write_ticker_info_file(TICKER_INFO_DICT)

    total_time = time.time() - global_start
    print(f"\n✓ Done: {len(tickers_dict)} tickers downloaded in {int(total_time // 60)}m {int(total_time % 60)}s")
    if failed_tickers:
        unique_failed = list(dict.fromkeys(failed_tickers))  # deduplicate
        print(f"✗ {len(unique_failed)} tickers had no data (likely delisted): {unique_failed[:20]}{'...' if len(unique_failed) > 20 else ''}")
        with open(os.path.join(DIR, "failed_tickers.txt"), "w") as f:
            f.write("\n".join(unique_failed))

    return tickers_dict

# ── TD Ameritrade (kept for compatibility, not actively maintained) ────────────

def tda_params(apikey, period_type="year", period=2, frequency_type="daily", frequency=1):
    return (
        ("apikey", apikey),
        ("periodType", period_type),
        ("period", period),
        ("frequencyType", frequency_type),
        ("frequency", frequency)
    )

def load_prices_from_tda(securities, api_key, info={}):
    print("*** Loading Stocks from TD Ameritrade ***")
    headers = {"Cache-Control": "no-cache"}
    params = tda_params(api_key)
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, sec in enumerate(securities):
        ticker = sec["ticker"]
        r_start = time.time()
        response = requests.get(TD_API % ticker, params=params, headers=headers)
        ticker_data = response.json()
        if ticker not in TICKER_INFO_DICT:
            load_ticker_info(ticker, TICKER_INFO_DICT)
            write_ticker_info_file(TICKER_INFO_DICT)
        ticker_data["industry"] = TICKER_INFO_DICT[ticker]["info"]["industry"]
        now = time.time()
        load_times.append(now - r_start)
        enrich_ticker_data(ticker_data, sec)
        tickers_dict[sec["ticker"]] = ticker_data
        error_text = f' Error {response.status_code}' if response.status_code != 200 else ''
        remaining_s = get_remaining_seconds(load_times, idx, len(list(securities)))
        print_data_progress(f"{ticker}{error_text}", idx + 1, len(list(securities)), now - start, remaining_s)
        if info.get("forceTDA"):
            sleep(0.4)

    write_price_history_file(tickers_dict)

# ── Entry point ───────────────────────────────────────────────────────────────

def save_data(source, securities, api_key, info={}):
    if source == "YAHOO":
        load_prices_from_yahoo(securities, info)
    elif source == "TD_AMERITRADE":
        load_prices_from_tda(securities, api_key, info)

def main(forceTDA=False, api_key=API_KEY):
    dataSource = DATA_SOURCE if not forceTDA else "TD_AMERITRADE"
    save_data(dataSource, SECURITIES, api_key, {"forceTDA": forceTDA})
    write_ticker_info_file(TICKER_INFO_DICT)

if __name__ == "__main__":
    main()
