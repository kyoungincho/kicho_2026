# SK Hynix Weekly Forecast → Dec 2026

Ensemble weekly price paths, walk-forward accuracy, and trigger-based action playbook.

## Run

```bash
pip install -r requirements.txt
python3 src/weekly_forecast_v2.py
```

Outputs in `output/`:

- `forecast_chart.png` — weekly path to Dec
- `probability_chart.png` — P(hit levels) by week
- `forecast_weekly.csv` — quantile + scenario medians
- `actions_weekly.csv` — week-by-week actions
- `accuracy_report.json` — metrics
- `REPORT.md` — Korean summary

## Models

- Block bootstrap / recovery bootstrap (DD≤−25%)
- Regime GBM with left-tail mixture
- Multi-asset factor (NVDA, SOXX, QQQ, Samsung, USDKRW)
- Soft fundamental anchor OU
- Bear / base / bull scenario blend
- Walk-forward inverse-error weights + interval calibration
