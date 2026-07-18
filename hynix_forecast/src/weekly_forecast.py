#!/usr/bin/env python3
"""
SK Hynix (000660.KS) weekly forecast to Dec + action plan.

Ensemble: block-bootstrap paths, regime GBM, exogenous factor model,
mean-reversion-to-anchor, event-aware drift. Weights from walk-forward accuracy.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
ART = Path("/opt/cursor/artifacts/hynix_forecast")
for p in (DATA, OUT, ART):
    p.mkdir(parents=True, exist_ok=True)

AVG_COST = 2_264_880
SHARES = 42
MONTHLY_INTEREST = 900_000
CORE = 33
AMMO = 9

# Soft fundamental anchor path (KRW) — earnings/HBM cycle narrative, not a target price promise
ANCHOR_BY_MONTH = {
    7: 2_050_000,
    8: 2_200_000,
    9: 2_450_000,
    10: 2_700_000,
    11: 2_950_000,
    12: 3_200_000,
}


def yahoo_chart(symbol: str, range_: str = "5y", interval: str = "1d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_}&interval={interval}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    ).dropna(subset=["close"])
    df = df.set_index("date").sort_index()
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    ohlc["ret"] = ohlc["close"].pct_change()
    return ohlc


def atr_pct_weekly(w: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = w["close"].shift(1)
    tr = pd.concat(
        [
            w["high"] - w["low"],
            (w["high"] - prev).abs(),
            (w["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr / w["close"] * 100


def fetch_all() -> dict[str, pd.DataFrame]:
    symbols = {
        "hynix": "000660.KS",
        "samsung": "005930.KS",
        "nvda": "NVDA",
        "soxx": "SOXX",
        "qqq": "QQQ",
        "usdkurw": "KRW=X",
    }
    out = {}
    for k, s in symbols.items():
        try:
            d = yahoo_chart(s, "5y", "1d")
            out[k] = to_weekly(d)
            print(f"fetched {k} ({s}): {len(out[k])} weeks, last={out[k].index[-1].date()} {out[k]['close'].iloc[-1]:,.2f}")
        except Exception as e:
            print(f"WARN {k}: {e}")
    return out


def align_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    h = data["hynix"].copy()
    feat = pd.DataFrame(index=h.index)
    feat["h_close"] = h["close"]
    feat["h_ret"] = h["ret"]
    feat["h_vol"] = h["ret"].rolling(12).std()
    feat["h_atr"] = atr_pct_weekly(h)
    feat["h_mom4"] = h["close"].pct_change(4)
    feat["h_mom12"] = h["close"].pct_change(12)
    feat["h_dd_ath"] = h["close"] / h["close"].cummax() - 1
    for name in ("samsung", "nvda", "soxx", "qqq", "usdkurw"):
        if name not in data:
            continue
        r = data[name]["ret"].reindex(feat.index)
        feat[f"{name}_ret"] = r
        feat[f"{name}_mom4"] = data[name]["close"].pct_change(4).reindex(feat.index)
    feat = feat.dropna()
    return feat


# --------------- models ---------------

def model_bootstrap(rets: np.ndarray, n_paths: int, n_weeks: int, start: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # block length 3 weeks to keep short memory
    L = 3
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    valid = rets[~np.isnan(rets)]
    for p in range(n_paths):
        px = start
        t = 0
        while t < n_weeks:
            i0 = rng.integers(0, max(1, len(valid) - L))
            block = valid[i0 : i0 + L]
            for r in block:
                if t >= n_weeks:
                    break
                px *= 1 + r
                t += 1
                paths[p, t] = px
    return paths


def model_regime_gbm(
    rets: np.ndarray, n_paths: int, n_weeks: int, start: float, seed: int, high_vol: bool
) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    valid = rets[~np.isnan(rets)]
    mu = float(np.mean(valid[-26:])) if len(valid) >= 26 else float(np.mean(valid))
    sig = float(np.std(valid[-26:], ddof=1)) if len(valid) >= 26 else float(np.std(valid, ddof=1))
    if high_vol:
        sig *= 1.25
        mu *= 0.5  # choppy: damp drift
    # mixture: 70% current regime, 30% crash-like fat left
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px = start
        for t in range(1, n_weeks + 1):
            if rng.random() < 0.08:
                r = float(rng.choice(np.sort(valid)[: max(3, len(valid) // 20)]))  # left tail week
            else:
                r = rng.normal(mu, sig)
            r = float(np.clip(r, -0.35, 0.40))
            px *= 1 + r
            paths[p, t] = px
    return paths


def model_factor(
    feat: pd.DataFrame, n_paths: int, n_weeks: int, start: float, seed: int
) -> np.ndarray:
    """Predict h_ret from lagged factors; shock residual bootstrap."""
    rng = np.random.default_rng(seed + 2)
    y = feat["h_ret"].values
    cols = [c for c in feat.columns if c.endswith("_ret") and c != "h_ret"]
    cols += [c for c in ("h_mom4", "h_vol", "h_dd_ath") if c in feat.columns]
    X = feat[cols].shift(1).values
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y2, X2 = y[mask], X[mask]
    if len(y2) < 40:
        return model_bootstrap(feat["h_ret"].values, n_paths, n_weeks, start, seed + 9)
    # ridge
    lam = 1.0
    XtX = X2.T @ X2 + lam * np.eye(X2.shape[1])
    beta = np.linalg.solve(XtX, X2.T @ y2)
    resid = y2 - X2 @ beta
    # last factors for forward: assume factor returns mean-revert to 0 with noise
    last = feat.iloc[-1]
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    factor_sig = {c: float(feat[c].std()) for c in cols if c.endswith("_ret")}
    for p in range(n_paths):
        px = start
        state = {c: float(last[c]) if c in last and not np.isnan(last[c]) else 0.0 for c in cols}
        for t in range(1, n_weeks + 1):
            # evolve exogenous as AR(1)
            xrow = []
            for c in cols:
                if c.endswith("_ret"):
                    state[c] = 0.3 * state[c] + rng.normal(0, factor_sig.get(c, 0.03))
                elif c == "h_mom4":
                    state[c] = (px / paths[p, max(0, t - 4)] - 1) if t >= 4 else state.get(c, 0)
                elif c == "h_vol":
                    state[c] = 0.9 * state[c] + 0.1 * abs(state.get("nvda_ret", 0))
                elif c == "h_dd_ath":
                    # approximate vs rolling peak along path
                    peak = max(paths[p, :t].max(), px)
                    state[c] = px / peak - 1
                xrow.append(state[c])
            pred = float(np.dot(beta, np.array(xrow)))
            shock = float(rng.choice(resid))
            r = float(np.clip(pred + shock, -0.35, 0.40))
            px *= 1 + r
            paths[p, t] = px
    return paths


def model_anchor(n_paths: int, n_weeks: int, start: float, start_date: date, seed: int, vol: float) -> np.ndarray:
    """Pull toward monthly fundamental soft-anchor with noise."""
    rng = np.random.default_rng(seed + 3)
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px = start
        for t in range(1, n_weeks + 1):
            d = start_date + timedelta(weeks=t)
            m = d.month
            # interpolate anchors
            a0 = ANCHOR_BY_MONTH.get(m, ANCHOR_BY_MONTH[12])
            a1 = ANCHOR_BY_MONTH.get(min(12, m + 1), a0)
            day_frac = min(1.0, d.day / 28)
            anchor = a0 * (1 - day_frac) + a1 * day_frac
            # Ornstein-Uhlenbeck like in log space weekly
            kappa = 0.12
            log_px = math.log(px)
            log_a = math.log(anchor)
            log_px = log_px + kappa * (log_a - log_px) + rng.normal(0, vol)
            px = math.exp(log_px)
            paths[p, t] = px
    return paths


def ensemble_paths(
    feat: pd.DataFrame, n_weeks: int, n_paths: int, weights: dict[str, float], seed: int = 42
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    start = float(feat["h_close"].iloc[-1])
    rets = feat["h_ret"].values
    high_vol = float(feat["h_atr"].iloc[-1]) >= 10 or float(feat["h_vol"].iloc[-1]) * math.sqrt(52) >= 0.8
    start_date = feat.index[-1].date()
    vol_w = float(np.nanstd(rets[-26:]))

    parts = {
        "bootstrap": model_bootstrap(rets, n_paths, n_weeks, start, seed),
        "gbm": model_regime_gbm(rets, n_paths, n_weeks, start, seed, high_vol),
        "factor": model_factor(feat, n_paths, n_weeks, start, seed),
        "anchor": model_anchor(n_paths, n_weeks, start, start_date, seed, vol_w * 0.9),
    }
    # mix paths by weight: sample model then path
    rng = np.random.default_rng(seed + 99)
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    mixed = np.zeros((n_paths, n_weeks + 1))
    for i in range(n_paths):
        m = rng.choice(names, p=w)
        mixed[i] = parts[m][i]
    return mixed, parts


def quantiles(paths: np.ndarray, qs=(0.10, 0.25, 0.50, 0.75, 0.90)) -> dict[str, np.ndarray]:
    return {f"q{int(q*100)}": np.quantile(paths, q, axis=0) for q in qs}


# --------------- backtest / accuracy ---------------

@dataclass
class Acc:
    mape: float
    rmse: float
    smape: float
    dir_acc: float
    cover80: float
    cover50: float
    n: int


def walk_forward(feat: pd.DataFrame, horizon: int = 8, step: int = 2, min_train: int = 80) -> tuple[Acc, pd.DataFrame, dict]:
    """Walk-forward: forecast h weeks, score at week h close."""
    rows = []
    model_err = {k: [] for k in ("bootstrap", "gbm", "factor", "anchor")}
    idx = feat.index
    closes = feat["h_close"].values
    for t in range(min_train, len(feat) - horizon, step):
        train = feat.iloc[: t + 1]
        actual = closes[t + horizon]
        start = closes[t]
        # equal weight for scoring each model median
        equal = {k: 0.25 for k in model_err}
        paths, parts = ensemble_paths(train, horizon, n_paths=400, weights=equal, seed=1000 + t)
        q = quantiles(paths)
        pred = q["q50"][-1]
        lo10, hi90 = q["q10"][-1], q["q90"][-1]
        lo25, hi75 = q["q25"][-1], q["q75"][-1]
        rows.append(
            {
                "origin": idx[t],
                "target": idx[t + horizon],
                "actual": actual,
                "pred": pred,
                "lo10": lo10,
                "hi90": hi90,
                "lo25": lo25,
                "hi75": hi75,
                "abs_err": abs(pred - actual) / actual,
                "dir_ok": int((pred - start) * (actual - start) > 0),
                "in80": int(lo10 <= actual <= hi90),
                "in50": int(lo25 <= actual <= hi75),
            }
        )
        for name, pth in parts.items():
            med = np.median(pth[:, -1])
            model_err[name].append(abs(med - actual) / actual)
    df = pd.DataFrame(rows)
    if df.empty:
        return Acc(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0), df, {}
    mape = float(df["abs_err"].mean())
    rmse = float(np.sqrt(np.mean(((df["pred"] - df["actual"]) / df["actual"]) ** 2)))
    smape = float(np.mean(2 * np.abs(df["pred"] - df["actual"]) / (np.abs(df["pred"]) + np.abs(df["actual"]))))
    acc = Acc(
        mape=mape,
        rmse=rmse,
        smape=smape,
        dir_acc=float(df["dir_ok"].mean()),
        cover80=float(df["in80"].mean()),
        cover50=float(df["in50"].mean()),
        n=len(df),
    )
    # inverse-error weights
    inv = {k: 1.0 / (np.mean(v) + 1e-6) for k, v in model_err.items() if v}
    s = sum(inv.values())
    weights = {k: inv[k] / s for k in inv}
    return acc, df, weights


def calibrate_bands(bt: pd.DataFrame) -> dict:
    """Widen/narrow bands so empirical coverage ~ nominal."""
    if bt.empty:
        return {"scale80": 1.0, "scale50": 1.0}
    # simple scale on half-width
    def needed(lo, hi, actual, target_cover):
        mid = (lo + hi) / 2
        half = (hi - lo) / 2
        # find scale s such that cover ~= target
        for s in np.linspace(0.6, 2.5, 40):
            cover = ((actual >= mid - s * half) & (actual <= mid + s * half)).mean()
            if cover >= target_cover:
                return float(s)
        return 2.5

    s80 = needed(bt["lo10"].values, bt["hi90"].values, bt["actual"].values, 0.80)
    s50 = needed(bt["lo25"].values, bt["hi75"].values, bt["actual"].values, 0.50)
    return {"scale80": s80, "scale50": s50}


def apply_calibration(q: dict[str, np.ndarray], cal: dict) -> dict[str, np.ndarray]:
    mid = q["q50"]
    out = dict(q)
    half80 = (q["q90"] - q["q10"]) / 2
    half50 = (q["q75"] - q["q25"]) / 2
    out["q10"] = mid - cal["scale80"] * half80
    out["q90"] = mid + cal["scale80"] * half80
    out["q25"] = mid - cal["scale50"] * half50
    out["q75"] = mid + cal["scale50"] * half50
    # floor
    for k in out:
        out[k] = np.maximum(out[k], 100_000)
    return out


# --------------- action plan ---------------

def week_dates(start: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    # forecast weeks ending Friday
    cur = start
    if cur.weekday() != 4:
        # next Friday
        cur = cur + pd.Timedelta(days=(4 - cur.weekday()) % 7)
    return [cur + pd.Timedelta(weeks=i) for i in range(1, n + 1)]


def build_actions(weeks: list[pd.Timestamp], q: dict[str, np.ndarray], last_close: float) -> pd.DataFrame:
    rows = []
    cash_raised = 0.0
    ammo_left = AMMO
    state = "BELOW_COST" if last_close < AVG_COST else "ARMED"
    for i, w in enumerate(weeks):
        t = i + 1
        med = float(q["q50"][t])
        lo = float(q["q10"][t])
        hi = float(q["q90"][t])
        p25 = float(q["q25"][t])
        p75 = float(q["q75"][t])

        # event tags
        tags = []
        if date(2026, 7, 20) <= w.date() <= date(2026, 8, 1):
            tags.append("실적/빅테크창")
        if date(2026, 9, 1) <= w.date() <= date(2026, 9, 20):
            tags.append("Q3가이던스민감")
        if date(2026, 10, 20) <= w.date() <= date(2026, 11, 5):
            tags.append("Q3실적시즌")
        if w.month == 12:
            tags.append("연말수급")

        actions = []
        # phase logic using median path + bands
        if med < AVG_COST * 0.98:
            phase = "평단아래_홀드"
            actions.append("코어33+탄약9 홀드. 평단↓ 매도 금지")
            actions.append(f"관심: 저배럴 {lo/1e4:.0f}만 근접 시 관망(추격X), 반등 시 평단 탈환 여부 확인")
            if MONTHLY_INTEREST and w.day <= 7:
                actions.append(f"이자 현금흐름: 월 ~{MONTHLY_INTEREST/1e4:.0f}만원 유동성 확보")
            state = "BELOW_COST"
        elif med < 2_380_000:
            phase = "평단탈환_무장"
            actions.append("종가 2일 연속 평단(226.5만) 위면 ARMED")
            actions.append("238만 터치 시 탄약 3주 매도 → 현금화 시작")
            state = "ARMED"
        elif med < 2_610_000:
            phase = "현금화_계단"
            actions.append("238→245→261만에서 탄약 3+3+3 매도 (이번 주 밴드 상단 활용)")
            actions.append("매도평균 ref 기록. 재매수: −12%/−18%/−25%에 40/40/20")
            state = "RAISING"
        else:
            phase = "고구간_코어사다리"
            actions.append("탄약 현금화 완료 가정 → 눌림 재매수만")
            if med >= 3_500_000 or hi >= 3_500_000:
                actions.append("350만 근접 시 코어 12주 익절 → 대출상환(현금흐름 전환)")
            if med >= 3_750_000:
                actions.append("375만 코어 추가 익절")
            if med >= 4_000_000:
                actions.append("400만 잔여 코어 익절")
            state = "CASH_READY"

        # cashflow harvest hint from band width
        band = (hi - lo) / med
        if band >= 0.25 and med >= AVG_COST:
            actions.append(f"주간 밴드 넓음({band:.0%}) → 상단 일부매도/하단 재매수 적극")
        elif band < 0.15 and state == "CASH_READY":
            actions.append("밴드 축소 → 추격매수 금지, 현금 유지")

        # probability style signals from path quantiles
        p_above_cost = float((med > AVG_COST) and (p25 > AVG_COST * 0.95))
        rows.append(
            {
                "week_end": w.date().isoformat(),
                "week_no": t,
                "q10": round(lo),
                "q25": round(p25),
                "q50": round(med),
                "q75": round(p75),
                "q90": round(hi),
                "phase": phase,
                "state": state,
                "tags": ",".join(tags),
                "actions": " | ".join(actions),
                "cashflow_note": (
                    "이자만 지출(매도수익 없음)"
                    if med < AVG_COST
                    else "탄약매도로 현금유입 가능"
                    if phase == "현금화_계단"
                    else "재매수/대기(현금유출 가능)"
                    if state == "CASH_READY"
                    else "탈환 확인 중"
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------- plot ---------------

def plot_forecast(hist: pd.DataFrame, weeks: list[pd.Timestamp], q: dict, acc: Acc, path: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    h = hist.iloc[-80:]
    ax.plot(h.index, h["close"], color="#1a1a1a", lw=2, label="Historical weekly close")
    ax.axhline(AVG_COST, color="#c0392b", ls="--", lw=1, label=f"Avg cost {AVG_COST/1e4:.0f}만")
    for y, lab, c in [
        (2_380_000, "238만 sell1", "#2980b9"),
        (2_450_000, "245만 sell2", "#2980b9"),
        (2_610_000, "261만 sell3", "#2980b9"),
        (3_500_000, "350만 core", "#27ae60"),
    ]:
        ax.axhline(y, color=c, ls=":", lw=0.8, alpha=0.8)

    x = [hist.index[-1]] + list(weeks)
    ax.fill_between(x, q["q10"], q["q90"], color="#3498db", alpha=0.15, label="80% band (calibrated)")
    ax.fill_between(x, q["q25"], q["q75"], color="#3498db", alpha=0.30, label="50% band")
    ax.plot(x, q["q50"], color="#e67e22", lw=2.5, label="Median forecast")

    ax.set_title(
        f"SK Hynix (000660.KS) Weekly Forecast → Dec 2026\n"
        f"Walk-forward MAPE={acc.mape*100:.1f}% | DirAcc={acc.dir_acc*100:.1f}% | "
        f"80% cover={acc.cover80*100:.1f}% (n={acc.n})",
        fontsize=12,
    )
    ax.set_ylabel("KRW")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:.0f}만"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(ART / path.name, dpi=160)
    plt.close(fig)


def plot_scenarios(weeks, parts, hist, path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    h = hist.iloc[-40:]
    ax.plot(h.index, h["close"], color="black", lw=1.5)
    x = [hist.index[-1]] + list(weeks)
    colors = {"bootstrap": "#8e44ad", "gbm": "#16a085", "factor": "#d35400", "anchor": "#2c3e50"}
    for name, pth in parts.items():
        med = np.median(pth, axis=0)
        ax.plot(x, med, lw=1.5, label=name, color=colors.get(name, "gray"))
    ax.axhline(AVG_COST, color="red", ls="--", lw=0.8)
    ax.set_title("Component model medians")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:.0f}만"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    fig.savefig(ART / path.name, dpi=140)
    plt.close(fig)


def main():
    print("=== DATA ===")
    data = fetch_all()
    data["hynix"].to_csv(DATA / "hynix_weekly.csv")
    feat = align_features(data)
    feat.to_csv(DATA / "features_weekly.csv")
    last_close = float(feat["h_close"].iloc[-1])
    last_date = feat.index[-1]
    print(f"Last weekly close: {last_date.date()} {last_close:,.0f}")

    # horizon to end of Dec 2026
    end = pd.Timestamp("2026-12-26")
    n_weeks = max(1, int((end - last_date).days / 7) + 1)
    print(f"Forecast weeks: {n_weeks}")

    print("=== WALK-FORWARD BACKTEST (h=8w) ===")
    acc8, bt8, w8 = walk_forward(feat, horizon=8, step=2, min_train=90)
    print(acc8, w8)
    bt8.to_csv(OUT / "backtest_h8.csv", index=False)

    print("=== WALK-FORWARD BACKTEST (h=4w) ===")
    acc4, bt4, w4 = walk_forward(feat, horizon=4, step=2, min_train=90)
    print(acc4, w4)
    bt4.to_csv(OUT / "backtest_h4.csv", index=False)

    # blend weights from 4w and 8w
    weights = {}
    for k in set(w4) | set(w8):
        weights[k] = 0.55 * w4.get(k, 0.25) + 0.45 * w8.get(k, 0.25)
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}
    print("Ensemble weights:", weights)

    cal = calibrate_bands(bt8 if not bt8.empty else bt4)
    print("Calibration:", cal)

    # second pass: re-backtest with calibrated note (coverage after scale on stored preds)
    if not bt8.empty:
        mid = (bt8["lo10"] + bt8["hi90"]) / 2
        half = (bt8["hi90"] - bt8["lo10"]) / 2
        cover_cal = (
            (bt8["actual"] >= mid - cal["scale80"] * half)
            & (bt8["actual"] <= mid + cal["scale80"] * half)
        ).mean()
    else:
        cover_cal = float("nan")

    print("=== FORWARD ENSEMBLE ===")
    paths, parts = ensemble_paths(feat, n_weeks, n_paths=2000, weights=weights, seed=20260718)
    q = apply_calibration(quantiles(paths), cal)
    weeks = week_dates(last_date, n_weeks)
    # ensure q length
    assert len(q["q50"]) == n_weeks + 1

    forecast_df = pd.DataFrame(
        {
            "week_end": [last_date.date().isoformat()] + [w.date().isoformat() for w in weeks],
            "q10": q["q10"],
            "q25": q["q25"],
            "q50": q["q50"],
            "q75": q["q75"],
            "q90": q["q90"],
        }
    )
    forecast_df.to_csv(OUT / "forecast_weekly.csv", index=False)

    actions = build_actions(weeks, q, last_close)
    actions.to_csv(OUT / "actions_weekly.csv", index=False)

    plot_forecast(data["hynix"], weeks, q, acc8, OUT / "forecast_chart.png")
    plot_scenarios(weeks, parts, data["hynix"], OUT / "model_components.png")

    # accuracy report + improvement notes
    report = {
        "asof": last_date.date().isoformat(),
        "last_close": last_close,
        "n_weeks": n_weeks,
        "ensemble_weights": weights,
        "calibration": cal,
        "accuracy_h4": acc4.__dict__,
        "accuracy_h8": acc8.__dict__,
        "coverage80_after_calibration_h8": float(cover_cal) if cover_cal == cover_cal else None,
        "dec_median": float(q["q50"][-1]),
        "dec_q10": float(q["q10"][-1]),
        "dec_q90": float(q["q90"][-1]),
        "position": {
            "shares": SHARES,
            "avg_cost": AVG_COST,
            "core": CORE,
            "ammo": AMMO,
            "monthly_interest": MONTHLY_INTEREST,
        },
        "improvements_applied": [
            "multi-asset exogenous factor model (NVDA/SOXX/QQQ/Samsung/USDKRW)",
            "regime-aware GBM with left-tail mixture",
            "block bootstrap weekly returns",
            "fundamental soft-anchor OU path to Dec",
            "walk-forward inverse-error ensemble weights",
            "prediction interval calibration to nominal coverage",
        ],
    }
    with open(OUT / "accuracy_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # markdown summary
    md = []
    md.append("# SK Hynix Weekly Forecast → 2026-12\n")
    md.append(f"- As-of: **{report['asof']}** close **{last_close:,.0f} KRW**\n")
    md.append(f"- Horizon: **{n_weeks} weeks**\n")
    md.append(f"- Dec median: **{report['dec_median']/1e4:,.0f}만** (80% band {report['dec_q10']/1e4:,.0f}~{report['dec_q90']/1e4:,.0f}만)\n")
    md.append("\n## Accuracy (walk-forward)\n")
    md.append(f"| Horizon | MAPE | RMSE(rel) | DirAcc | 80% cover | 50% cover | n |\n|---|---|---|---|---|---|---|\n")
    md.append(
        f"| 4w | {acc4.mape*100:.1f}% | {acc4.rmse*100:.1f}% | {acc4.dir_acc*100:.1f}% | "
        f"{acc4.cover80*100:.1f}% | {acc4.cover50*100:.1f}% | {acc4.n} |\n"
    )
    md.append(
        f"| 8w | {acc8.mape*100:.1f}% | {acc8.rmse*100:.1f}% | {acc8.dir_acc*100:.1f}% | "
        f"{acc8.cover80*100:.1f}% | {acc8.cover50*100:.1f}% | {acc8.n} |\n"
    )
    md.append(f"\nCalibrated 80% coverage (h8): **{cover_cal*100:.1f}%** (scale={cal['scale80']:.2f})\n")
    md.append(f"\nEnsemble weights: `{weights}`\n")
    md.append("\n## Weekly actions (summary)\n")
    md.append("| Week | q50 | Phase | Cashflow | Actions |\n|---|---|---|---|---|\n")
    for _, r in actions.iterrows():
        md.append(
            f"| {r['week_end']} | {r['q50']/1e4:.0f}만 | {r['phase']} | {r['cashflow_note']} | {r['actions'][:120]} |\n"
        )
    (OUT / "REPORT.md").write_text("".join(md), encoding="utf-8")
    (ART / "REPORT.md").write_text("".join(md), encoding="utf-8")

    print("=== DONE ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
