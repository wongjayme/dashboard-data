"""
Fetch fundamental data for all tickers in snapshot.json via finvizfinance.

Fields fetched per ticker:
  - eps_this_y_pct   : EPS Growth This Year (%)
  - eps_next_y_pct   : EPS Growth Next Year (%)
  - eps_next_5y_pct  : EPS Growth Next 5 Years (%)
  - eps_qoq_pct      : EPS Growth Quarter over Quarter (%)
  - sales_qoq_pct    : Sales Growth Quarter over Quarter (%)
  - profit_margin_pct: Profit Margin (%)
  - earnings_date    : Next earnings date, ISO format (e.g. '2026-07-22')

Output: data/fundamentals.json

Run:
  python scripts/fetch_fundamentals.py [--snapshot-path data/snapshot.json] [--out-dir data]

GitHub Actions:
  python scripts/fetch_fundamentals.py --snapshot-path data/snapshot.json --out-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date, datetime

import pandas as pd
import requests
from curl_cffi import requests as curl_requests
import finvizfinance.util as _futil
from finvizfinance import quote as _fvquote
from finvizfinance.quote import finvizfinance

# ---------------------------------------------------------------------------
# Patch 1: curl_cffi session — bypasses Cloudflare TLS fingerprinting
# ---------------------------------------------------------------------------

_futil.session = curl_requests.Session(impersonate="chrome110")

# ---------------------------------------------------------------------------
# Patch 2: fix ticker_fundament — Finviz renamed "quote-links" to
# "quote-header_categories" and added a 5th link, shifting Exchange to index 4
# ---------------------------------------------------------------------------

def _patched_ticker_fundament(self, raw=True, output_format="dict"):
    if output_format not in ["dict", "series"]:
        raise ValueError(
            "Invalid output format '{}'. Possible choice: {}".format(
                output_format, ["dict", "series"]
            )
        )
    fundament_info = {}

    fundament_info["Company"] = self.soup.find(
        "h2", class_="quote-header_ticker-wrapper_company"
    ).text.strip()

    quote_links = self.soup.find("div", class_="quote-header_categories")
    links = quote_links.find_all("a")
    fundament_info["Sector"]   = links[0].text
    fundament_info["Industry"] = links[1].text
    fundament_info["Country"]  = links[2].text
    fundament_info["Exchange"] = links[4].text  # index shifted: new cap-size link at [3]

    # Finviz now splits the fundamentals into multiple separate
    # <table class="snapshot-table2"> blocks (confirmed: 6 on the quote
    # page as of July 2026), instead of one big table. The old .find()
    # only grabbed the first one, silently dropping EPS/growth/margin/
    # valuation stats that live in the other 5 tables.
    fundament_tables = self.soup.find_all("table", class_="snapshot-table2")

    for fundament_table in fundament_tables:
        rows = fundament_table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            cols = [i.text for i in cols]
            fundament_info = self._parse_column(cols, raw, fundament_info)
    self.info["fundament"] = fundament_info

    if output_format == "dict":
        return fundament_info
    return pd.DataFrame.from_dict(fundament_info, orient="index", columns=["Stat"])

_fvquote.finvizfinance.ticker_fundament = _patched_ticker_fundament

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DELAY_BETWEEN_REQUESTS = 0.5   # seconds — keeps Finviz happy
MAX_RETRIES            = 3     # max retry attempts per ticker on transient errors
RETRY_BASE_DELAY       = 2.0   # seconds — exponential backoff base (2, 4, 8)
RETRY_STATUS_CODES     = {429, 502, 503, 504}  # HTTP codes worth retrying

FIELD_MAP = {
    "eps_this_y_pct":    "EPS this Y",
    "eps_next_y_pct":    "EPS next Y Percentage",
    "eps_next_5y_pct":   "EPS next 5Y",
    "eps_qoq_pct":       "EPS Q/Q",
    "sales_qoq_pct":     "Sales Q/Q",
    "profit_margin_pct": "Profit Margin",
    "fwd_pe":            "Forward P/E",
    "ps_ratio":          "P/S",
    "peg_ratio":         "PEG",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pct(val) -> float | None:
    """Parse a Finviz percentage string like '13.66%' or '-5.20%' into a float."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace("%", "").replace(",", "")
        if s in ("-", "", "N/A"):
            return None
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None


def parse_float(val) -> float | None:
    """Parse a plain Finviz float string like '24.5' or '1.8' into a float."""
    if val is None:
        return None
    try:
        s = str(val).strip().replace(",", "")
        if s in ("-", "", "N/A"):
            return None
        return round(float(s), 2)
    except (ValueError, TypeError):
        return None


def parse_earnings_date(val) -> str | None:
    """Parse Finviz's 'Earnings' field (e.g. 'Jun 18 AMC') into an ISO date.

    Finviz gives month/day with no year, so the year is inferred: if the
    resulting date has already passed, roll forward to next year.
    """
    if not val:
        return None
    s = str(val).strip()
    if s in ("-", "", "N/A"):
        return None
    m = re.match(r'([A-Za-z]{3})\s+(\d{1,2})', s)
    if not m:
        return None
    mon_str, day_str = m.groups()
    try:
        month = datetime.strptime(mon_str, "%b").month
        day = int(day_str)
    except ValueError:
        return None
    today = date.today()
    try:
        d = date(today.year, month, day)
    except ValueError:
        return None
    if d < today:
        try:
            d = date(today.year + 1, month, day)
        except ValueError:
            return None
    return d.isoformat()


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a transient server error worth retrying."""
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else None
        return code in RETRY_STATUS_CODES
    msg = str(exc).lower()
    return any(str(c) in msg for c in RETRY_STATUS_CODES)


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetch and parse fundamental fields for a single ticker, with retry/backoff."""
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            stock     = finvizfinance(ticker)
            fundament = stock.ticker_fundament()
            earnings_date = parse_earnings_date(fundament.get("Earnings"))
            return {
                "eps_this_y_pct":    parse_pct(fundament.get("EPS this Y")),
                "eps_next_y_pct":    parse_pct(fundament.get("EPS next Y Percentage")),
                "eps_next_5y_pct":   parse_pct(fundament.get("EPS next 5Y")),
                "eps_qoq_pct":       parse_pct(fundament.get("EPS Q/Q")),
                "sales_qoq_pct":     parse_pct(fundament.get("Sales Q/Q")),
                "profit_margin_pct": parse_pct(fundament.get("Profit Margin")),
                "fwd_pe":            parse_float(fundament.get("Forward P/E")),
                "ps_ratio":          parse_float(fundament.get("P/S")),
                "peg_ratio":         parse_float(fundament.get("PEG")),
                "earnings_date":     earnings_date,
            }

        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  RETRY [{ticker}] attempt {attempt + 1}/{MAX_RETRIES} "
                      f"after {wait:.0f}s — {e}")
                time.sleep(wait)
            else:
                break

    print(f"  ERROR [{ticker}]: {last_exc}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", default="data/snapshot.json", help="Path to snapshot.json")
    parser.add_argument("--out-dir",       default="data",               help="Output directory")
    args = parser.parse_args()

    print(f"Loading snapshot: {args.snapshot_path}")
    with open(args.snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    tickers = []
    for rows in snapshot.get("by_industry", {}).values():
        for row in rows:
            t = row.get("ticker")
            if t:
                tickers.append(t)

    tickers = sorted(set(tickers))
    total   = len(tickers)
    print(f"Tickers to fetch: {total}\n")

    results  = {}
    failed   = []
    start    = time.time()

    for i, ticker in enumerate(tickers, 1):
        data = fetch_fundamentals(ticker)
        if data:
            results[ticker] = data
            print(f"  [{i:04d}/{total}] {ticker:<8} "
                  f"EPS_TY={str(data['eps_this_y_pct']):<8} "
                  f"EPS_NY={str(data['eps_next_y_pct']):<8} "
                  f"EPS_5Y={str(data['eps_next_5y_pct']):<8} "
                  f"EPS_QQ={str(data['eps_qoq_pct']):<8} "
                  f"SLS_QQ={str(data['sales_qoq_pct']):<8} "
                  f"PM={str(data['profit_margin_pct']):<8} "
                  f"FWD_PE={str(data['fwd_pe']):<8} "
                  f"PS={str(data['ps_ratio']):<8} "
                  f"PEG={str(data['peg_ratio']):<8} "
                  f"EARN={str(data['earnings_date'])}")
        else:
            failed.append(ticker)
            print(f"  [{i:04d}/{total}] {ticker:<8} FAILED")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    elapsed = time.time() - start
    print(f"\nDone: {len(results)} succeeded, {len(failed)} failed in {elapsed/60:.1f} min")
    if failed:
        print(f"Failed tickers: {failed}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "fundamentals.json")
    output = {
        "built_at":     datetime.utcnow().isoformat() + "Z",
        "ticker_count": len(results),
        "fundamentals": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
