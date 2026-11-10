#!/usr/bin/env python
# update_stocks.py — Weekly refresh of ticker_info.json
# Runs every Sunday via update_stocks.yml
#
# What it does:
#   - Loops through all tickers in the existing ticker_info.json
#   - Re-fetches ALL metadata fields: sector, industry, marketCap, floatShares,
#     shortPercentOfFloat, revenueGrowth, fiftyTwoWeekHigh, fiftyTwoWeekLow, exchange
#   - This ensures market data (MarketCap, Float, RevenueGrowth etc.) stays fresh weekly
#   - Also picks up any new tickers that appeared in the NASDAQ list
#
# Speed: ~2-3 hours for 6800 tickers (individual calls — acceptable for weekly run)

import json
import random
import requests
import yfinance as yf
from pathlib import Path
from io import StringIO
from time import sleep

TICKER_INFO_FILE = Path(__file__).parent / "ticker_info.json"
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"

YFINANCE_EXCHANGE_MAP = {
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NCM": "NASDAQ",
    "NGM": "NASDAQ",
    "ASE": "AMEX",
    "ARC": "NYSE ARCA",
    "BTS": "BATS",
}

def exchange_from_yfinance(code):
    return YFINANCE_EXCHANGE_MAP.get(code, code if code else "n/a")

def safe_float(d, key):
    try:
        v = d.get(key) if hasattr(d, "get") else getattr(d, key, None)
        return float(v) if v is not None else None
    except (TypeError, ValueError, AttributeError):
        return None

def fetch_ticker_info(symbol, max_retries=3):
    """
    Fetch all metadata for a ticker using fast_info + .info.
    Returns a dict with all fields, or None on total failure.
    """
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(symbol)
            result = {
                "industry": "n/a", "sector": "n/a",
                "marketCap": None, "floatShares": None,
                "shortPercentOfFloat": None, "revenueGrowth": None,
                "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
                "exchange": "n/a",
            }

            # fast_info — reliable for price/market data
            try:
                fi = t.fast_info
                result["marketCap"]        = safe_float(fi, "marketCap")
                result["fiftyTwoWeekHigh"] = safe_float(fi, "yearHigh")
                result["fiftyTwoWeekLow"]  = safe_float(fi, "yearLow")
                result["floatShares"]      = safe_float(fi, "shares")
            except Exception:
                pass

            # .info — for sector, industry, short, revenue, exchange
            try:
                info = t.info
                if not info or "symbol" not in info:
                    raise ValueError("Empty info")
                result["industry"]            = info.get("industry") or "n/a"
                result["sector"]              = info.get("sector")   or "n/a"
                result["shortPercentOfFloat"] = (
                    safe_float(info, "shortPercentOfFloat") or
                    safe_float(info, "shortRatio")
                )
                result["revenueGrowth"]       = safe_float(info, "revenueGrowth")
                result["exchange"]            = exchange_from_yfinance(info.get("exchange", ""))
                if result["marketCap"] is None:
                    result["marketCap"]       = safe_float(info, "marketCap")
            except Exception:
                pass

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt + 1}/{max_retries} for {symbol} in {wait:.1f}s: {e}")
                sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {symbol}: {e}")

    return None

def get_nasdaq_tickers():
    """Download current NASDAQ ticker list via HTTPS."""
    r = requests.get(NASDAQ_URL, timeout=60)
    r.raise_for_status()
    lines = r.text.splitlines()
    tickers = set()
    for line in lines[1:]:
        vals = line.split("|")
        if len(vals) < 8:
            continue
        symbol = vals[1].strip()
        etf    = vals[5].strip()
        test   = vals[7].strip()
        if symbol and etf == "N" and test == "N" and symbol.isalpha() and len(symbol) <= 5:
            tickers.add(symbol)
    print(f"NASDAQ list: {len(tickers)} tickers")
    return tickers

def main():
    # Load existing cache
    if TICKER_INFO_FILE.exists():
        with open(TICKER_INFO_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded existing cache: {len(cache)} tickers")
    else:
        cache = {}
        print("No existing cache — starting fresh")

    # Get current NASDAQ tickers
    nasdaq_tickers = get_nasdaq_tickers()

    # Union: refresh all existing + add new ones
    all_tickers = sorted(nasdaq_tickers | set(cache.keys()))
    print(f"Total to process: {len(all_tickers)} tickers")

    updated = 0
    failed  = 0

    for idx, symbol in enumerate(all_tickers):
        info = fetch_ticker_info(symbol)

        if info is not None:
            cache[symbol] = {"info": info}
            updated += 1
        else:
            failed += 1
            # Keep existing data if we have it
            if symbol not in cache:
                cache[symbol] = {"info": {
                    "industry": "n/a", "sector": "n/a",
                    "marketCap": None, "floatShares": None,
                    "shortPercentOfFloat": None, "revenueGrowth": None,
                    "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None,
                    "exchange": "n/a",
                }}

        # Progress every 100 tickers
        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(all_tickers)}] Updated: {updated} | Failed: {failed}")
            with open(TICKER_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

        # Throttle — 1-2s between calls (weekly run, no rush)
        sleep(random.uniform(1, 2))
        # Longer pause every 50 tickers
        if (idx + 1) % 50 == 0:
            sleep(3)

    # Final save
    with open(TICKER_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    print(f"\n✓ Done: {updated} updated, {failed} failed, {len(cache)} total in cache")

if __name__ == "__main__":
    main()
