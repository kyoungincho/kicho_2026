#!/usr/bin/env python3
"""
SK Hynix weekly forecast v2 — drawdown-conditioned ensemble + scenarios + trigger playbook.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import date, timedelta
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
DATA, OUT = ROOT / "data", ROOT / "output"
ART = Path("/opt/cursor/artifacts/hynix_forecast")
for p in (DATA, OUT, ART):
    p.mkdir(parents=True, exist_ok=True)

AVG_COST = 2_264_880
SHARES, CORE, AMMO = 42, 33, 9
MONTHLY_INTEREST = 900_000
SELL_LADDER = [2_380_000, 2_450_000, 2_610_000]
CORE_LADDER = [3_500_000, 3_750_000, 4_000_000]
REBUY_DROP = (0.12, 0.18, 0.25)
REBUY_FRAC = (0.40, 0.40, 0.20)

# Soft anchors (narrative, low backtest weight but used in bull/base scenarios)
ANCHOR = {7: 2.05e6, 8: 2.25e6, 9: 2.55e6, 10: 2.85e6, 11: 3.15e6, 12: 3.45e6}


def yahoo(symbol: str, range_: str = "5y") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(None),
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
            "volume": q["volume"],
        }
    ).dropna(subset=["close"]).set_index("date").sort_index()
    return df


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    w["ret"] = w["close"].pct_change()
    return w


def atr_pct(w: pd.DataFrame, n=14) -> pd.Series:
    prev = w["close"].shift(1)
    tr = pd.concat(
        [w["high"] - w["low"], (w["high"] - prev).abs(), (w["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean() / w["close"] * 100


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = {}
    for k, s in {
        "hynix": "000660.KS",
        "samsung": "005930.KS",
        "nvda": "NVDA",
        "soxx": "SOXX",
        "qqq": "QQQ",
        "usdkurw": "KRW=X",
    }.items():
        raw[k] = weekly(yahoo(s))
        print(f"{k}: {raw[k].index[-1].date()} {raw[k]['close'].iloc[-1]:,.2f}")
    h = raw["hynix"]
    feat = pd.DataFrame(index=h.index)
    feat["h_close"] = h["close"]
    feat["h_ret"] = h["ret"]
    feat["h_vol"] = h["ret"].rolling(12).std()
    feat["h_atr"] = atr_pct(h)
    feat["h_mom4"] = h["close"].pct_change(4)
    feat["h_mom12"] = h["close"].pct_change(12)
    feat["h_dd"] = h["close"] / h["close"].cummax() - 1
    for name in ("samsung", "nvda", "soxx", "qqq", "usdkurw"):
        feat[f"{name}_ret"] = raw[name]["ret"].reindex(feat.index)
        feat[f"{name}_mom4"] = raw[name]["close"].pct_change(4).reindex(feat.index)
    feat = feat.dropna()
    h.to_csv(DATA / "hynix_weekly.csv")
    feat.to_csv(DATA / "features_weekly.csv")
    return h, feat


# ---- path generators ----

def block_bootstrap(rets, n_paths, n_weeks, start, seed, block=3, tilt=0.0):
    rng = np.random.default_rng(seed)
    valid = rets[~np.isnan(rets)]
    # tilt: overweight positive weeks after deep drawdown
    w = np.ones(len(valid))
    if tilt > 0:
        w = np.where(valid > 0, 1 + tilt, 1.0)
        w = w / w.sum()
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px, t = start, 0
        while t < n_weeks:
            if tilt > 0:
                i0 = rng.choice(max(1, len(valid) - block), p=w[: max(1, len(valid) - block)] / w[: max(1, len(valid) - block)].sum())
            else:
                i0 = rng.integers(0, max(1, len(valid) - block))
            for r in valid[i0 : i0 + block]:
                if t >= n_weeks:
                    break
                px *= 1 + float(r)
                t += 1
                paths[p, t] = px
    return paths


def recovery_bootstrap(closes, rets, n_paths, n_weeks, start, seed):
    """Sample forward paths from historical windows that started at DD<=-25%."""
    rng = np.random.default_rng(seed)
    dd = closes / np.maximum.accumulate(closes) - 1
    starts = [i for i in range(len(closes) - n_weeks - 1) if dd[i] <= -0.25]
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    if len(starts) < 3:
        return block_bootstrap(rets, n_paths, n_weeks, start, seed, tilt=0.8)
    for p in range(n_paths):
        i0 = int(rng.choice(starts))
        px = start
        for t in range(1, n_weeks + 1):
            r = rets[i0 + t]
            if np.isnan(r):
                r = float(rng.choice(rets[~np.isnan(rets)]))
            # scale shock slightly toward current vol
            px *= 1 + float(r)
            paths[p, t] = px
    return paths


def gbm_mix(rets, n_paths, n_weeks, start, seed, mu_scale=1.0, sig_scale=1.0):
    rng = np.random.default_rng(seed)
    valid = rets[~np.isnan(rets)]
    mu = float(np.mean(valid[-26:])) * mu_scale
    sig = float(np.std(valid[-26:], ddof=1)) * sig_scale
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px = start
        for t in range(1, n_weeks + 1):
            if rng.random() < 0.07:
                r = float(np.quantile(valid, rng.uniform(0.02, 0.12)))
            else:
                r = rng.normal(mu, sig)
            px *= 1 + float(np.clip(r, -0.35, 0.40))
            paths[p, t] = px
    return paths


def factor_model(feat, n_paths, n_weeks, start, seed):
    rng = np.random.default_rng(seed)
    y = feat["h_ret"].values
    cols = [c for c in feat.columns if c.endswith("_ret") and c != "h_ret"]
    cols += [c for c in ("h_mom4", "h_vol", "h_dd") if c in feat.columns]
    X = feat[cols].shift(1).values
    mask = ~np.isnan(y) & ~np.isnan(X).any(1)
    y2, X2 = y[mask], X[mask]
    lam = 1.0
    beta = np.linalg.solve(X2.T @ X2 + lam * np.eye(X2.shape[1]), X2.T @ y2)
    resid = y2 - X2 @ beta
    last = feat.iloc[-1]
    sig = {c: float(feat[c].std()) for c in cols if c.endswith("_ret")}
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px = start
        state = {c: float(last[c]) if c in last and pd.notna(last[c]) else 0.0 for c in cols}
        for t in range(1, n_weeks + 1):
            xrow = []
            for c in cols:
                if c.endswith("_ret"):
                    state[c] = 0.25 * state[c] + rng.normal(0, sig.get(c, 0.03))
                elif c == "h_mom4":
                    state[c] = px / paths[p, max(0, t - 4)] - 1 if t >= 4 else state[c]
                elif c == "h_vol":
                    state[c] = 0.85 * state[c] + 0.15 * abs(state.get("nvda_ret", 0))
                elif c == "h_dd":
                    peak = max(paths[p, :t].max(), px)
                    state[c] = px / peak - 1
                xrow.append(state[c])
            r = float(np.clip(np.dot(beta, xrow) + rng.choice(resid), -0.35, 0.40))
            px *= 1 + r
            paths[p, t] = px
    return paths


def anchor_ou(n_paths, n_weeks, start, start_date, seed, vol, kappa=0.15, anchor_scale=1.0):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_weeks + 1))
    paths[:, 0] = start
    for p in range(n_paths):
        px = start
        for t in range(1, n_weeks + 1):
            d = start_date + timedelta(weeks=t)
            a0 = ANCHOR.get(d.month, ANCHOR[12]) * anchor_scale
            a1 = ANCHOR.get(min(12, d.month + 1), a0) * anchor_scale
            anchor = a0 * (1 - min(1, d.day / 28)) + a1 * min(1, d.day / 28)
            log_px = math.log(px) + kappa * (math.log(anchor) - math.log(px)) + rng.normal(0, vol)
            px = math.exp(log_px)
            paths[p, t] = px
    return paths


def mix_parts(parts: dict[str, np.ndarray], weights: dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = list(weights)
    w = np.array([weights[n] for n in names], float)
    w /= w.sum()
    n_paths, cols = next(iter(parts.values())).shape
    out = np.zeros((n_paths, cols))
    for i in range(n_paths):
        out[i] = parts[rng.choice(names, p=w)][i]
    return out


def qtile(paths, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return {f"q{int(q*100)}": np.quantile(paths, q, 0) for q in qs}


# ---- backtest ----

@dataclass
class Acc:
    mape: float
    rmse: float
    smape: float
    dir_acc: float
    cover80: float
    cover50: float
    n: int


def eval_paths(pred_med, lo10, hi90, lo25, hi75, actual, start) -> dict:
    return {
        "abs_err": abs(pred_med - actual) / actual,
        "sq": ((pred_med - actual) / actual) ** 2,
        "smape": 2 * abs(pred_med - actual) / (abs(pred_med) + abs(actual)),
        "dir_ok": int((pred_med - start) * (actual - start) > 0),
        "in80": int(lo10 <= actual <= hi90),
        "in50": int(lo25 <= actual <= hi75),
    }


def make_parts(feat_train, n_weeks, n_paths, seed):
    start = float(feat_train["h_close"].iloc[-1])
    rets = feat_train["h_ret"].values
    closes = feat_train["h_close"].values
    sd = feat_train.index[-1].date()
    vol = float(np.nanstd(rets[-26:]))
    dd = float(feat_train["h_dd"].iloc[-1])
    tilt = 0.9 if dd <= -0.25 else 0.0
    return {
        "bootstrap": block_bootstrap(rets, n_paths, n_weeks, start, seed, tilt=tilt),
        "recovery": recovery_bootstrap(closes, rets, n_paths, n_weeks, start, seed + 1),
        "gbm": gbm_mix(rets, n_paths, n_weeks, start, seed + 2, mu_scale=0.7 if dd > -0.2 else 1.1, sig_scale=1.15),
        "factor": factor_model(feat_train, n_paths, n_weeks, start, seed + 3),
        "anchor": anchor_ou(n_paths, n_weeks, start, sd, seed + 4, vol * 0.9, kappa=0.12),
    }


def walk_forward(feat, horizon=8, step=2, min_train=90, recent_only: pd.Timestamp | None = None):
    rows, err = [], {k: [] for k in ("bootstrap", "recovery", "gbm", "factor", "anchor")}
    closes = feat["h_close"].values
    idx = feat.index
    for t in range(min_train, len(feat) - horizon, step):
        if recent_only is not None and idx[t] < recent_only:
            continue
        train = feat.iloc[: t + 1]
        parts = make_parts(train, horizon, 300, seed=2000 + t)
        equal = {k: 1 / len(parts) for k in parts}
        paths = mix_parts(parts, equal, seed=3000 + t)
        q = qtile(paths)
        e = eval_paths(q["q50"][-1], q["q10"][-1], q["q90"][-1], q["q25"][-1], q["q75"][-1], closes[t + horizon], closes[t])
        e.update({"origin": idx[t], "actual": closes[t + horizon], "pred": q["q50"][-1]})
        rows.append(e)
        for name, pth in parts.items():
            err[name].append(abs(np.median(pth[:, -1]) - closes[t + horizon]) / closes[t + horizon])
    df = pd.DataFrame(rows)
    if df.empty:
        return Acc(*[np.nan] * 6, 0), df, {k: 0.2 for k in err}
    acc = Acc(
        mape=float(df["abs_err"].mean()),
        rmse=float(np.sqrt(df["sq"].mean())),
        smape=float(df["smape"].mean()),
        dir_acc=float(df["dir_ok"].mean()),
        cover80=float(df["in80"].mean()),
        cover50=float(df["in50"].mean()),
        n=len(df),
    )
    inv = {k: 1 / (np.mean(v) + 1e-6) for k, v in err.items() if v}
    s = sum(inv.values())
    weights = {k: inv[k] / s for k in inv}
    return acc, df, weights


def calibrate(bt: pd.DataFrame):
    if bt.empty:
        return {"scale80": 1.0, "scale50": 1.0}

    def scale_for(lo, hi, actual, target):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for s in np.linspace(0.5, 2.5, 41):
            if ((actual >= mid - s * half) & (actual <= mid + s * half)).mean() >= target:
                return float(s)
        return 2.5

    return {
        "scale80": scale_for(bt["pred"] * 0 + bt.get("in80", 0), bt["pred"], bt["actual"], 0.8)
        if False
        else scale_for(
            bt["pred"] - (bt["pred"] * 0 + 1),  # placeholder replaced below
            bt["pred"] + 1,
            bt["actual"],
            0.8,
        ),
    }


def calibrate_from_parts(bt_rows_path: Path | None, bt: pd.DataFrame, feat, horizon, weights):
    # Recompute lo/hi on a sample of origins for calibration — use stored pred errors empirically
    if bt.empty:
        return {"scale80": 1.0, "scale50": 1.0}
    # Use residual distribution: band half-width proxy via quantile of abs errors
    ae = bt["abs_err"].values
    # For reporting we already have cover from equal-weight; scale so cover→80/50 using expanding half from mape
    # Simpler: inflate/deflate based on cover gap
    s80 = 1.0
    if bt["in80"].mean() < 0.78:
        s80 = 1.0 + (0.80 - bt["in80"].mean()) * 2
    elif bt["in80"].mean() > 0.90:
        s80 = max(0.7, 1.0 - (bt["in80"].mean() - 0.80))
    s50 = 1.0
    if bt["in50"].mean() < 0.45:
        s50 = 1.0 + (0.50 - bt["in50"].mean()) * 2
    elif bt["in50"].mean() > 0.65:
        s50 = max(0.7, 1.0 - (bt["in50"].mean() - 0.50))
    return {"scale80": float(s80), "scale50": float(s50), "abs_err_p80": float(np.quantile(ae, 0.8))}


def apply_cal(q, cal):
    mid = q["q50"]
    h80 = (q["q90"] - q["q10"]) / 2 * cal["scale80"]
    h50 = (q["q75"] - q["q25"]) / 2 * cal["scale50"]
    out = {
        "q50": mid,
        "q10": np.maximum(mid - h80, 1e5),
        "q90": mid + h80,
        "q25": np.maximum(mid - h50, 1e5),
        "q75": mid + h50,
    }
    return out


# ---- scenarios (explicit) ----

def scenario_paths(feat, n_weeks, n_paths_each=800):
    start = float(feat["h_close"].iloc[-1])
    rets = feat["h_ret"].values
    closes = feat["h_close"].values
    sd = feat.index[-1].date()
    vol = float(np.nanstd(rets[-26:]))
    bear = gbm_mix(rets, n_paths_each, n_weeks, start, 11, mu_scale=-0.5, sig_scale=1.3)
    # also mix crash continuation
    bear = 0.6 * bear + 0.4 * block_bootstrap(rets, n_paths_each, n_weeks, start, 12, tilt=-0.5)
    # fix: can't average paths like that meaningfully — regenerate
    bear = mix_parts(
        {
            "g": gbm_mix(rets, n_paths_each, n_weeks, start, 11, mu_scale=-0.3, sig_scale=1.35),
            "b": block_bootstrap(rets, n_paths_each, n_weeks, start, 12, tilt=-0.6),
        },
        {"g": 0.5, "b": 0.5},
        13,
    )
    base = mix_parts(
        {
            "r": recovery_bootstrap(closes, rets, n_paths_each, n_weeks, start, 21),
            "f": factor_model(feat, n_paths_each, n_weeks, start, 22),
            "g": gbm_mix(rets, n_paths_each, n_weeks, start, 23, mu_scale=1.0, sig_scale=1.1),
        },
        {"r": 0.45, "f": 0.35, "g": 0.20},
        24,
    )
    bull = mix_parts(
        {
            "a": anchor_ou(n_paths_each, n_weeks, start, sd, 31, vol * 0.75, kappa=0.18, anchor_scale=1.05),
            "r": recovery_bootstrap(closes, rets, n_paths_each, n_weeks, start, 32),
            "g": gbm_mix(rets, n_paths_each, n_weeks, start, 33, mu_scale=1.6, sig_scale=1.0),
        },
        {"a": 0.45, "r": 0.35, "g": 0.20},
        34,
    )
    return {"bear": bear, "base": base, "bull": bull}


def scenario_probs_from_dd(dd: float, atr: float) -> dict[str, float]:
    # deep DD + high ATR → more weight to base recovery & still material bear
    if dd <= -0.35 and atr >= 10:
        return {"bear": 0.28, "base": 0.47, "bull": 0.25}
    if dd <= -0.25:
        return {"bear": 0.25, "base": 0.50, "bull": 0.25}
    return {"bear": 0.30, "base": 0.45, "bull": 0.25}


def blend_scenarios(scen, probs, n_paths=2500, seed=99):
    rng = np.random.default_rng(seed)
    names = list(probs)
    p = np.array([probs[n] for n in names])
    cols = scen["base"].shape[1]
    out = np.zeros((n_paths, cols))
    for i in range(n_paths):
        name = rng.choice(names, p=p)
        j = rng.integers(0, scen[name].shape[0])
        out[i] = scen[name][j]
    return out


# ---- actions ----

def week_ends(last: pd.Timestamp, n: int):
    cur = last if last.weekday() == 4 else last + pd.Timedelta(days=(4 - last.weekday()) % 7)
    if cur <= last:
        cur = last + pd.Timedelta(weeks=1)
        # align friday
        cur = cur + pd.Timedelta(days=(4 - cur.weekday()) % 7)
    return [last + pd.Timedelta(weeks=i) for i in range(1, n + 1)]


def build_playbook(weeks, q, scen_medians, probs, paths, last_close) -> pd.DataFrame:
    rows = []
    n = len(weeks)
    for i, w in enumerate(weeks):
        t = i + 1
        med, lo, hi = float(q["q50"][t]), float(q["q10"][t]), float(q["q90"][t])
        p25, p75 = float(q["q25"][t]), float(q["q75"][t])
        bear_m = float(scen_medians["bear"][t])
        base_m = float(scen_medians["base"][t])
        bull_m = float(scen_medians["bull"][t])
        # path probabilities this week
        col = paths[:, t]
        p_cost = float((col >= AVG_COST).mean())
        p_s1 = float((col >= SELL_LADDER[0]).mean())
        p_s2 = float((col >= SELL_LADDER[1]).mean())
        p_s3 = float((col >= SELL_LADDER[2]).mean())
        p_350 = float((col >= CORE_LADDER[0]).mean())
        p_deep = float((col <= last_close * 0.90).mean())

        tags = []
        wd = w.date() if hasattr(w, "date") else w
        if date(2026, 7, 20) <= wd <= date(2026, 8, 2):
            tags.append("실적·빅테크")
        if date(2026, 9, 1) <= wd <= date(2026, 9, 25):
            tags.append("Q3민감")
        if date(2026, 10, 20) <= wd <= date(2026, 11, 10):
            tags.append("Q3실적")
        if wd.month == 12:
            tags.append("연말")

        # TRIGGER playbook (price-based, not median-only)
        triggers = []
        triggers.append(f"IF ≥238만: 탄약 3주 매도 (누적현금↑) [P≈{p_s1:.0%}]")
        triggers.append(f"IF ≥245만: 탄약 +3주 매도 [P≈{p_s2:.0%}]")
        triggers.append(f"IF ≥261만: 탄약 +3주 매도 [P≈{p_s3:.0%}]")
        triggers.append("IF 매도평균 대비 −12/−18/−25%: 현금 40/40/20 재매수")
        triggers.append(f"IF ≥350만: 코어 12주 익절→대출상환 [P≈{p_350:.0%}]")
        triggers.append(f"IF ≤{last_close*0.9/1e4:.0f}만(추가 −10%): 신규매수 금지·홀드 / 여유현금 있을 때만 3차 대기")

        # default stance this week
        if p_cost < 0.35:
            stance = "방어홀드"
            primary = "평단 아래 구간 확률 우위 → 매도 없음. 이자 유동성만 확보. 트리거는 예약."
            cashflow = f"순유출: 이자 ~주 {MONTHLY_INTEREST/4/1e4:.0f}만원 수준"
        elif p_s1 >= 0.30:
            stance = "현금화준비"
            primary = "상단 터치 확률 유의 → 238부터 탄약 매도 지정가 필수."
            cashflow = "조건부 유입: 탄약 매도 시 주당 238~261만"
        else:
            stance = "탈환감시"
            primary = "평단 탈환 시도 구간. 종가 2일 연속 226.5만↑ 확인 후 매도계단 가동."
            cashflow = "유입 대기 / 이자 유출"

        if "실적·빅테크" in tags or "Q3실적" in tags:
            primary += " | 이벤트창: 매도는 허용, 재매수는 발표 익일 종가 후."

        rows.append(
            {
                "week_end": wd.isoformat(),
                "week_no": t,
                "q10": int(lo),
                "q25": int(p25),
                "q50": int(med),
                "q75": int(p75),
                "q90": int(hi),
                "bear": int(bear_m),
                "base": int(base_m),
                "bull": int(bull_m),
                "P_ge_cost": round(p_cost, 3),
                "P_ge_238": round(p_s1, 3),
                "P_ge_245": round(p_s2, 3),
                "P_ge_261": round(p_s3, 3),
                "P_ge_350": round(p_350, 3),
                "P_m10": round(p_deep, 3),
                "stance": stance,
                "tags": ",".join(tags),
                "primary_action": primary,
                "triggers": " || ".join(triggers),
                "cashflow": cashflow,
            }
        )
    return pd.DataFrame(rows)


def plot_main(h, weeks, q, scen_med, acc, probs, out: Path):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    hist = h.iloc[-90:]
    ax.plot(hist.index, hist["close"], color="#111", lw=2, label="Weekly close")
    ax.axhline(AVG_COST, color="#c0392b", ls="--", lw=1.2, label="Avg cost")
    for y in SELL_LADDER:
        ax.axhline(y, color="#2980b9", ls=":", lw=0.9, alpha=0.9)
    ax.axhline(CORE_LADDER[0], color="#27ae60", ls=":", lw=1)

    x = [h.index[-1]] + list(weeks)
    ax.fill_between(x, q["q10"], q["q90"], color="#3498db", alpha=0.14, label="80% band")
    ax.fill_between(x, q["q25"], q["q75"], color="#3498db", alpha=0.28, label="50% band")
    ax.plot(x, q["q50"], color="#e67e22", lw=2.6, label="Ensemble median")
    ax.plot(x, scen_med["bear"], color="#7f8c8d", lw=1.5, ls="--", label=f"Bear ({probs['bear']:.0%})")
    ax.plot(x, scen_med["base"], color="#2c3e50", lw=1.8, label=f"Base ({probs['base']:.0%})")
    ax.plot(x, scen_med["bull"], color="#27ae60", lw=1.5, ls="--", label=f"Bull ({probs['bull']:.0%})")

    ax.set_title(
        f"SK Hynix Weekly Path to Dec 2026  |  MAPE(8w)={acc.mape*100:.1f}%  Dir={acc.dir_acc*100:.1f}%  "
        f"80%cover={acc.cover80*100:.1f}%",
        fontsize=11,
    )
    ax.set_ylabel("KRW")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:.0f}만"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    fig.savefig(ART / out.name, dpi=170)
    plt.close()


def plot_prob(weeks, play: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = pd.to_datetime(play["week_end"])
    ax.plot(x, play["P_ge_cost"], label="P(≥ cost)", lw=2)
    ax.plot(x, play["P_ge_238"], label="P(≥ 238만)", lw=1.5)
    ax.plot(x, play["P_ge_350"], label="P(≥ 350만)", lw=1.5)
    ax.set_ylim(0, 1)
    ax.set_title("Path probabilities by week (ensemble)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(ART / out.name, dpi=150)
    plt.close()


def main():
    print("=== LOAD ===")
    h, feat = load()
    last_close = float(feat["h_close"].iloc[-1])
    last_date = feat.index[-1]
    end = pd.Timestamp("2026-12-26")
    n_weeks = max(1, int((end - last_date).days / 7) + 1)
    print(f"last={last_date.date()} {last_close:,.0f} weeks={n_weeks}")

    print("=== BACKTEST full ===")
    acc8, bt8, w8 = walk_forward(feat, 8, 2, 90)
    acc4, bt4, w4 = walk_forward(feat, 4, 2, 90)
    print("h8", acc8, w8)
    print("h4", acc4, w4)
    bt8.to_csv(OUT / "backtest_h8.csv", index=False)
    bt4.to_csv(OUT / "backtest_h4.csv", index=False)

    print("=== BACKTEST recent 1y ===")
    acc8r, bt8r, w8r = walk_forward(feat, 8, 1, 90, recent_only=last_date - pd.Timedelta(days=400))
    acc4r, bt4r, w4r = walk_forward(feat, 4, 1, 90, recent_only=last_date - pd.Timedelta(days=400))
    print("h8 recent", acc8r, w8r)
    print("h4 recent", acc4r)

    # weights: prefer recent regime for forward
    weights = {}
    for k in set(w8) | set(w8r) | set(w4):
        weights[k] = 0.35 * w8.get(k, 0.2) + 0.40 * w8r.get(k, 0.2) + 0.25 * w4.get(k, 0.2)
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}
    print("weights", weights)

    cal = calibrate_from_parts(None, bt8, feat, 8, weights)
    # recent cover adjust
    if not bt8r.empty:
        if bt8r["in80"].mean() < 0.75:
            cal["scale80"] *= 1.15
    print("cal", cal)

    print("=== SCENARIOS + ENSEMBLE ===")
    scen = scenario_paths(feat, n_weeks)
    dd, atr = float(feat["h_dd"].iloc[-1]), float(feat["h_atr"].iloc[-1])
    probs = scenario_probs_from_dd(dd, atr)
    print("dd", dd, "atr", atr, "probs", probs)

    # also accuracy-weighted statistical blend
    parts = make_parts(feat, n_weeks, 1200, seed=20260718)
    stat_paths = mix_parts(parts, weights, seed=777)
    scen_paths = blend_scenarios(scen, probs, n_paths=2500, seed=888)
    # final: 55% scenario blend (forward narrative+recovery), 45% pure statistical weights
    rng = np.random.default_rng(999)
    final = np.vstack(
        [
            scen_paths[rng.choice(len(scen_paths), 1400, replace=False)],
            stat_paths[rng.choice(len(stat_paths), 1100, replace=False)],
        ]
    )
    q = apply_cal(qtile(final), cal)
    scen_med = {k: np.median(v, 0) for k, v in scen.items()}

    weeks = [last_date + pd.Timedelta(weeks=i) for i in range(1, n_weeks + 1)]
    # snap to Friday labels
    weeks = [w + pd.Timedelta(days=(4 - w.weekday()) % 7) for w in weeks]

    fdf = pd.DataFrame(
        {
            "week_end": [last_date.date().isoformat()] + [w.date().isoformat() for w in weeks],
            "q10": q["q10"],
            "q25": q["q25"],
            "q50": q["q50"],
            "q75": q["q75"],
            "q90": q["q90"],
            "bear": scen_med["bear"],
            "base": scen_med["base"],
            "bull": scen_med["bull"],
        }
    )
    fdf.to_csv(OUT / "forecast_weekly.csv", index=False)

    play = build_playbook(weeks, q, scen_med, probs, final, last_close)
    play.to_csv(OUT / "actions_weekly.csv", index=False)

    plot_main(h, weeks, q, scen_med, acc8r if acc8r.n else acc8, probs, OUT / "forecast_chart.png")
    plot_prob(weeks, play, OUT / "probability_chart.png")

    # component chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(h.iloc[-40:].index, h.iloc[-40:]["close"], color="black", lw=1.5)
    x = [h.index[-1]] + list(weeks)
    for name, col in [("bear", "#7f8c8d"), ("base", "#2c3e50"), ("bull", "#27ae60")]:
        ax.plot(x, scen_med[name], label=name, color=col, lw=2)
    ax.plot(x, q["q50"], label="ensemble", color="#e67e22", lw=2)
    ax.axhline(AVG_COST, color="red", ls="--")
    ax.legend()
    ax.set_title("Scenario medians")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e4:.0f}만"))
    fig.tight_layout()
    fig.savefig(OUT / "model_components.png", dpi=150)
    fig.savefig(ART / "model_components.png", dpi=150)
    plt.close()

    # hit probs by month end
    milestones = {}
    for label, thr in [("cost", AVG_COST), ("238", 2380000), ("261", 2610000), ("350", 3500000), ("400", 4000000)]:
        milestones[label] = {
            "by_end_Aug": float((final[:, min(6, n_weeks)] >= thr).mean()),
            "by_end_Oct": float((final[:, min(15, n_weeks)] >= thr).mean()),
            "by_end_Dec": float((final[:, -1] >= thr).mean()),
            "ever_by_Dec": float((final[:, 1:].max(1) >= thr).mean()),
        }

    report = {
        "asof": last_date.date().isoformat(),
        "last_close": last_close,
        "drawdown_from_ath": dd,
        "atr_pct": atr,
        "n_weeks": n_weeks,
        "scenario_probs": probs,
        "model_weights": weights,
        "calibration": cal,
        "accuracy": {
            "h4_full": asdict(acc4),
            "h8_full": asdict(acc8),
            "h4_recent": asdict(acc4r),
            "h8_recent": asdict(acc8r),
        },
        "december": {
            "q10": float(q["q10"][-1]),
            "q50": float(q["q50"][-1]),
            "q90": float(q["q90"][-1]),
            "bear": float(scen_med["bear"][-1]),
            "base": float(scen_med["base"][-1]),
            "bull": float(scen_med["bull"][-1]),
        },
        "milestones": milestones,
        "position": {
            "shares": SHARES,
            "avg_cost": AVG_COST,
            "core": CORE,
            "ammo": AMMO,
            "monthly_interest": MONTHLY_INTEREST,
            "sell_ladder": SELL_LADDER,
            "rebuy": {"drops": REBUY_DROP, "fracs": REBUY_FRAC},
            "core_ladder": CORE_LADDER,
        },
        "improvements": [
            "recovery bootstrap conditioned on DD<=-25%",
            "tilt bootstrap after deep drawdown",
            "explicit bear/base/bull scenarios with DD/ATR priors",
            "55/45 blend of scenario paths and accuracy-weighted statistical models",
            "recent-1y walk-forward weighting",
            "interval calibration from coverage gap",
            "week-level trigger probabilities from full path distribution",
        ],
    }
    (OUT / "accuracy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Korean markdown report
    lines = []
    lines.append("# SK하이닉스 주간 전망 (→ 2026-12) & 액션 플랜\n\n")
    lines.append(f"- 기준: **{report['asof']}** 주간종가 **{last_close:,.0f}원** (ATH 대비 {dd:.1%})\n")
    lines.append(f"- 시나리오 사전확률: bear {probs['bear']:.0%} / base {probs['base']:.0%} / bull {probs['bull']:.0%}\n")
    lines.append(
        f"- 12월 말: 중앙 **{report['december']['q50']/1e4:,.0f}만** | "
        f"80%밴드 {report['december']['q10']/1e4:,.0f}~{report['december']['q90']/1e4:,.0f}만 | "
        f"bear/base/bull 중앙 {report['december']['bear']/1e4:,.0f} / {report['december']['base']/1e4:,.0f} / {report['december']['bull']/1e4:,.0f}만\n"
    )
    lines.append("\n## 정확도 (Walk-forward)\n\n")
    lines.append("| 구간 | MAPE | RMSE | 방향정확도 | 80%커버 | 50%커버 | n |\n|---|---|---|---|---|---|---|\n")
    for name, a in [
        ("4주-전체", acc4),
        ("8주-전체", acc8),
        ("4주-최근1년", acc4r),
        ("8주-최근1년", acc8r),
    ]:
        lines.append(
            f"| {name} | {a.mape*100:.1f}% | {a.rmse*100:.1f}% | {a.dir_acc*100:.1f}% | "
            f"{a.cover80*100:.1f}% | {a.cover50*100:.1f}% | {a.n} |\n"
        )
    lines.append("\n### 마일스톤 도달 확률 (경로 시뮬레이션)\n\n")
    lines.append("| 가격 | 8월말 | 10월말 | 12월말 | 연내 한번이라도 |\n|---|---|---|---|---|\n")
    for k, lab in [("cost", "평단 226.5만"), ("238", "238만"), ("261", "261만"), ("350", "350만"), ("400", "400만")]:
        m = milestones[k]
        lines.append(
            f"| {lab} | {m['by_end_Aug']:.0%} | {m['by_end_Oct']:.0%} | {m['by_end_Dec']:.0%} | {m['ever_by_Dec']:.0%} |\n"
        )
    lines.append("\n## 포지션 규칙 (고정)\n\n")
    lines.append(f"- 코어 **{CORE}주** / 탄약 **{AMMO}주** / 평단 **{AVG_COST:,}**\n")
    lines.append("- 현금화: 238·245·261만에서 3+3+3\n")
    lines.append("- 재매수: 매도평균 대비 −12/−18/−25%에 현금 40/40/20 (저점 탄약 남기기)\n")
    lines.append("- 코어익절: 350×12(대출상환) → 375×12 → 400 잔여\n")
    lines.append("- 월 이자 ~90만원: 매도 현금 발생 시 이자+원금 우선\n")
    lines.append("\n## 주차별 액션\n\n")
    lines.append("| 주 | 중앙 | base | bull | P≥평단 | P≥238 | 스탠스 | 기본액션 | 현금흐름 |\n|---|---|---|---|---|---|---|---|---|\n")
    for _, r in play.iterrows():
        lines.append(
            f"| {r['week_end']} | {r['q50']/1e4:.0f} | {r['base']/1e4:.0f} | {r['bull']/1e4:.0f} | "
            f"{r['P_ge_cost']:.0%} | {r['P_ge_238']:.0%} | {r['stance']} | {r['primary_action'][:80]} | {r['cashflow']} |\n"
        )
    lines.append("\n### 트리거 (매주 공통, 지정가 예약)\n\n")
    lines.append(play.iloc[0]["triggers"].replace(" || ", "\n- ") + "\n")
    lines.append("\n![](forecast_chart.png)\n\n![](probability_chart.png)\n")

    (OUT / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    (ART / "REPORT.md").write_text("".join(lines), encoding="utf-8")
    # copy charts already in ART via plot
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DONE")


if __name__ == "__main__":
    main()
