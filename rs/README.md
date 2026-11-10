# RS log (in-house)

Our own copy of the IBD-style Relative Strength log, so the dashboard no longer
depends on `Fred6725/rs-log` being updated.

The code is copied unchanged from
[Fred6725/relative-strength](https://github.com/Fred6725/relative-strength)
(commit `fdecfc5`, Apache-2.0, see `LICENSE`), which is itself a fork of
[skyte/relative-strength](https://github.com/skyte/relative-strength) via
[maximbelyayev/relative-strength](https://github.com/maximbelyayev/relative-strength).
The only change is `EXIT_WAIT_FOR_ENTER: false` in `config.yaml`.

## What runs

| Workflow | When | What it does |
|---|---|---|
| `.github/workflows/rs_log.yml` | Mon–Fri 23:30 UTC (after the US close) | Downloads ~6,000 tickers from Yahoo, ranks them, commits `rs/output/*.csv` and the `ticker_info.json` cache (~15–25 min) |
| `.github/workflows/build.yml` | Automatically when the RS log finishes (plus the nightly backstop) | Builds `data/` for the dashboard from `rs/output/` |
| `.github/workflows/rs_ticker_info.yml` | Sunday 00:00 UTC | Refreshes sector / industry / market cap / float / short interest for every ticker (~3 h) |

All three can also be started by hand from the Actions tab ("Run workflow").

## Outputs (same files and columns as Fred's rs-log)

| File | Content |
|---|---|
| `rs/output/rs_stocks.csv` | All stocks — the file the dashboard reads |
| `rs/output/rs_stocks_1.csv` / `rs_stocks_2.csv` | Percentile 50–99 / 0–49 (small enough for GitHub to render) |
| `rs/output/rs_industries.csv` | Industry rankings — also read by the dashboard |
| `rs/output/RSRATING.csv` | Percentile thresholds in TradingView Pine Seeds format |

Raw URL for Google Sheets or anything else:
`https://raw.githubusercontent.com/wongjayme/dashboard-data/main/rs/output/rs_stocks.csv`

## Calculation

```
RS score = 0.4 × P3 + 0.2 × P6 + 0.2 × P9 + 0.2 × P12   (P3 = last 63 trading days, etc.)
Relative Strength = (1 + RS score stock) / (1 + RS score SPY) × 100
Percentile = rank of Relative Strength across all stocks, 0–99
```

## Running locally

```bash
cd rs
pip install -r requirements.txt
python relative-strength.py true
```

Must be run from inside `rs/` (the script reads `config.yaml` from the current folder).
