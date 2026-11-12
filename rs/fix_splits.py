"""
Repair stock splits that Yahoo hasn't applied to its price history yet.

Runs between the price download (rs_data) and the ranking (rs_ranking) in
.github/workflows/rs_log.yml. Not part of Fred's original code.

Why: on the day of a split, Yahoo often keeps serving the OLD unadjusted
prices for every earlier day for a few hours. A 1-for-24 reverse split then
looks like a 2,300% one-day gain (SLXN and GOVX ranked 97-99 on 2026-11-11
instead of near the bottom), and a 2-for-1 forward split looks like a 50% crash.

How:
  1. Find stocks whose close rose 45%+ or fell 31%+ from one session to the
     next in the last few sessions.
  2. Ask Yahoo for each one's split history (only these few stocks).
  3. If a recent split explains the jump, the earlier prices are unadjusted:
     divide every price before the split date by the split ratio (and multiply
     volume by it), exactly what Yahoo itself does once it catches up.
Big moves with no split behind them are real and are left alone.
Any failure here is logged and skipped, so it can never stop the RS run.
"""
import datetime as dt
import json
import math
import os
import sys
import time

DIR = os.path.dirname(os.path.realpath(__file__))
PRICE_DATA_FILE = os.path.join(DIR, "data", "price_history.json")

JUMP = 1.45            # a session-to-session move this large (up 45%+ or down 31%+) is a
                       # candidate: catches 2-for-1 and 3-for-2 splits as well as reverse splits
LOOKBACK_SESSIONS = 3  # unadjusted history only lasts hours, so the last few sessions suffice
SPLIT_MAX_AGE_DAYS = 14

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = dt.timezone(dt.timedelta(hours=-5))


def session_date(ts):
    return dt.datetime.fromtimestamp(ts, NY).date()


def get_splits(ticker):
    """[(date, ratio)] for the last SPLIT_MAX_AGE_DAYS. ratio = new shares per old
    share, as yfinance reports it: 2.0 for 2-for-1, 0.041667 for 1-for-24."""
    import yfinance as yf
    s = yf.Ticker(ticker.replace(".", "-")).splits
    if s is None or len(s) == 0:
        return []
    cutoff = dt.date.today() - dt.timedelta(days=SPLIT_MAX_AGE_DAYS)
    out = []
    for idx, ratio in s.items():
        d = idx.date() if hasattr(idx, "date") else idx
        if d >= cutoff and ratio and ratio > 0:
            out.append((d, float(ratio)))
    return out


def candidates(price_data):
    found = []
    for ticker, entry in price_data.items():
        candles = entry.get("candles") if isinstance(entry, dict) else None
        if not candles or len(candles) < 2:
            continue
        start = max(1, len(candles) - LOOKBACK_SESSIONS)
        for i in range(start, len(candles)):
            a, b = candles[i - 1].get("close"), candles[i].get("close")
            if a and b and a > 0 and b > 0 and (b / a >= JUMP or b / a <= 1 / JUMP):
                found.append(ticker)
                break
    return found


def adjust(candles, split_date, ratio):
    """Return True if the pre-split candles were unadjusted and have been fixed."""
    k = next((i for i, c in enumerate(candles) if session_date(c["datetime"]) >= split_date), None)
    if k is None or k == 0:
        return False
    before, after = candles[k - 1].get("close"), candles[k].get("close")
    if not before or not after or before <= 0 or after <= 0:
        return False
    jump = after / before
    expected = 1 / ratio  # what an unadjusted split looks like (24x for 1-for-24, 0.5x for 2-for-1)
    size = abs(math.log(expected))
    if size < math.log(1.3):
        return False      # tiny split ratio: can't tell from an ordinary move
    # Unadjusted only if the jump is much nearer the split's size than to no change at all
    # (on a log scale, within half the split's size). An already-adjusted split shows
    # little jump at that date, so it is never adjusted twice.
    if abs(math.log(jump) - math.log(expected)) > size / 2:
        return False
    for c in candles[:k]:
        for f in ("open", "high", "low", "close"):
            if c.get(f) is not None:
                c[f] = c[f] / ratio
        if c.get("volume") is not None:
            c["volume"] = c["volume"] * ratio
    return True


def main(split_lookup=get_splits):
    if not os.path.exists(PRICE_DATA_FILE):
        print(f"fix_splits: {PRICE_DATA_FILE} not found - skipping")
        return 0
    with open(PRICE_DATA_FILE, encoding="utf8") as fp:
        price_data = json.load(fp)

    cands = candidates(price_data)
    print(f"fix_splits: {len(cands)} stock(s) with a big one-day move in the last {LOOKBACK_SESSIONS} sessions")
    fixed = 0
    for t in cands:
        try:
            splits = split_lookup(t)
        except Exception as e:
            print(f"  {t}: split lookup failed ({e}) - left as is")
            continue
        did = False
        for d, ratio in sorted(splits):
            if adjust(price_data[t]["candles"], d, ratio):
                label = f"1-for-{round(1 / ratio)}" if ratio < 1 else f"{ratio:g}-for-1"
                print(f"  {t}: {label} split on {d} not yet applied by Yahoo - earlier prices adjusted")
                did = True
        if did:
            fixed += 1
        else:
            print(f"  {t}: no unadjusted split - real move, left as is")
        time.sleep(0.3)

    if fixed:
        with open(PRICE_DATA_FILE, "w", encoding="utf8") as fp:
            json.dump(price_data, fp, ensure_ascii=False)
    print(f"fix_splits: adjusted {fixed} stock(s)")
    return fixed


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never block the RS run
        print(f"fix_splits: skipped after error: {e}")
    sys.exit(0)
