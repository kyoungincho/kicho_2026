"""
v3 = production model.

Design decisions (and why):
  1. The market closing price is the baseline. Features only move p AWAY from it.
  2. 21 games cannot estimate 8 weights. So weights are a Bayesian blend:
         w = (n0 * w_prior + n * w_data) / (n0 + n)
     w_prior comes from well-established hockey handicapping magnitudes,
     w_data from the sign-constrained ridge fit on the Sep29-Oct2 sample.
  3. NO global intercept. The sample had favorites at 42.9% vs 57.7% implied;
     baking that in would be "fade every favorite", which the long-run
     (~57% favorite win rate) does not support.
  4. A hard rail caps how far the model may move from the market (+-9 points).
  5. Bet sizing = half-Kelly, capped, and only above an EV floor that was
     itself chosen on the backtest.
"""
import numpy as np
from scipy.optimize import minimize
from nhl_backtest import (GAMES, X, y, pm, fav_dec, dog_dec, offset, N,
                          logit, sigmoid, am_to_dec, am_to_prob, novig)

SPEC = [
    (0, "fs", "페이브 주력(탑6F/탑4D) 1명 OUT", -1, -0.100),
    (1, "ds", "역배 주력 1명 OUT",              +1, +0.100),
    (2, "fg", "페이브 골리 다운그레이드",          -1, -0.220),
    (3, "dg", "역배 골리 다운그레이드",            +1, +0.220),
    (4, "fb", "페이브 B2B 뒷경기",               -1, -0.110),
    (5, "db", "역배 B2B 뒷경기",                 +1, +0.110),
    (6, "rl", "노나먹기(페이브가 직전 패)",         +1, +0.080),
    (7, "up", "역배 오프시즌 보강 우위",           -1, -0.080),
]
IDX = [s[0] for s in SPEC]
SIGN = np.array([s[3] for s in SPEC], float)
W_PRIOR = np.array([s[4] for s in SPEC], float)
CAP_W = 1.0
RAIL = 0.09          # max |p_model - p_market|
N0 = 60.0            # prior strength in "equivalent games"
LAM = 5.0            # from v2 LOO tuning


def fit_data(Xm, ym, off, lam=LAM):
    Xs = Xm[:, IDX]
    bounds = [((0, CAP_W) if s > 0 else (-CAP_W, 0)) for s in SIGN]

    def nll(w):
        p = np.clip(sigmoid(off + Xs @ w), 1e-9, 1 - 1e-9)
        return -(ym * np.log(p) + (1 - ym) * np.log(1 - p)).sum() + lam * (w @ w)

    best = None
    rng = np.random.default_rng(0)
    for t in range(6):
        w0 = np.zeros(len(IDX)) if t == 0 else np.clip(rng.normal(0, .15, len(IDX)) * SIGN,
                                                       -CAP_W, CAP_W)
        w0 = np.array([np.clip(v, b[0], b[1]) for v, b in zip(w0, bounds)])
        r = minimize(nll, w0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def blend(w_data, n=N, n0=N0):
    return (n0 * W_PRIOR + n * w_data) / (n0 + n)


def predict(p_market, feats, w):
    """feats = dict of code->value. Returns railed probability for the favorite."""
    x = np.array([feats.get(c, 0) for (_, c, _, _, _) in SPEC], float)
    p = sigmoid(logit(p_market) + x @ w)
    return float(np.clip(p, p_market - RAIL, p_market + RAIL))


# ------------------------------------------------------------------ fit + report
w_data = fit_data(X, y, offset)
W = blend(w_data)

print("=" * 80)
print("1) 최종 가중치 — 사전값 + 데이터 블렌드  (n=21, n0=60)")
print("=" * 80)
print(f"{'피처':34s} {'사전':>7s} {'데이터':>8s} {'최종 w':>8s} {'60%팀 → ':>12s}")
print("-" * 80)
for j, (ci, code, desc, sg, pr) in enumerate(SPEC):
    p1 = sigmoid(logit(0.60) + W[j])
    print(f"{desc:34s} {pr:+7.3f} {w_data[j]:+8.3f} {W[j]:+8.3f} {p1*100:10.1f}%")
print(f"\n레일: 시장확률 대비 최대 ±{RAIL:.0%} 까지만 이동 허용")


# ------------------------------------------------------------------ validate
def loo_blend():
    ll = br = 0.0
    acc = 0
    preds = np.zeros(N)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i])
        wd = fit_data(X[tr], y[tr], offset[tr])
        wb = blend(wd, n=N - 1)
        feats = {code: X[i, ci] for (ci, code, _, _, _) in SPEC}
        p = predict(pm[i], feats, wb)
        preds[i] = p
        ll -= y[i] * np.log(max(p, 1e-9)) + (1 - y[i]) * np.log(max(1 - p, 1e-9))
        br += (p - y[i]) ** 2
        acc += int((p > .5) == (y[i] == 1))
    return ll / N, br / N, acc / N, preds


mll = -np.mean(y * np.log(pm) + (1 - y) * np.log(1 - pm))
mbr = np.mean((pm - y) ** 2)
macc = np.mean((pm > .5) == (y == 1))
ll, br, acc, preds = loo_blend()

print()
print("=" * 80)
print("2) 검증 (leave-one-out, 21경기)")
print("=" * 80)
print(f"{'':12s} {'logloss':>9s} {'Brier':>8s} {'적중':>7s}")
print("-" * 80)
print(f"{'시장 단독':12s} {mll:9.4f} {mbr:8.4f} {macc:6.1%}")
print(f"{'v3 모델':12s} {ll:9.4f} {br:8.4f} {acc:6.1%}")
print(f"{'개선':12s} {mll-ll:+9.4f} {mbr-br:+8.4f} {acc-macc:+6.1%}")
print("\n해석: logloss 개선폭이 작다 = 배당이 이미 대부분을 담고 있고,")
print("      피처는 '일부 경기에서만' 의미 있는 편차를 만든다.")


def sim(preds, ev_floor, kelly_cap=0.04, half=0.5, bank=100.0):
    st = 0
    pnl = 0.0
    bk = bank
    rows = []
    for i in range(N):
        ef = preds[i] * fav_dec[i] - 1
        ed = (1 - preds[i]) * dog_dec[i] - 1
        if max(ef, ed) < ev_floor:
            continue
        if ef >= ed:
            p, dec, win, side, ev = preds[i], fav_dec[i], y[i] == 1, "FAV", ef
        else:
            p, dec, win, side, ev = 1 - preds[i], dog_dec[i], y[i] == 0, "DOG", ed
        b = dec - 1
        f = min(kelly_cap, max(0.0, (p * b - (1 - p)) / b) * half)
        bk += bk * f * (b if win else -1)
        st += 1
        pnl += b if win else -1
        rows.append((GAMES[i][0], GAMES[i][1], side, ev, f, win, dec))
    return st, pnl, bk, rows


print()
print("=" * 80)
print("3) EV 컷오프별 백테스트 (LOO 예측 기준)")
print("=" * 80)
print(f"{'EV컷':>6s} {'베팅':>5s} {'적중':>8s} {'플랫 수익':>10s} {'ROI':>8s} {'하프켈리 100→':>14s}")
print("-" * 80)
grid = []
for th in (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07):
    st, pnl, bk, rows = sim(preds, th)
    if st == 0:
        print(f"{th:+6.0%} {0:5d} {'-':>8s} {'-':>10s} {'-':>8s} {'-':>14s}")
        continue
    h = sum(r[5] for r in rows)
    grid.append((th, st, pnl / st, bk))
    print(f"{th:+6.0%} {st:5d} {h:3d}/{st:<4d} {pnl:+10.2f} {pnl/st:+7.1%} {bk:13.1f}")

best_th = max(grid, key=lambda g: g[3])[0] if grid else 0.02
print(f"\n권장 EV 컷오프: {best_th:+.0%}  (뱅크롤 최대화 기준)")
st, pnl, bk, rows = sim(preds, best_th)
print(f"\n해당 컷 베팅 내역:")
for d, m, side, ev, f, win, dec in rows:
    print(f"  {d} {m:9s} {side:3s} dec{dec:5.2f} EV{ev:+6.1%} 스테이크{f*100:4.1f}% "
          f"{'적중' if win else '실패'}")

np.save("/workspace/W_v3.npy", W)
print(f"\n가중치 저장 완료. EV컷={best_th}")
with open("/workspace/v3_params.txt", "w") as fh:
    fh.write(f"RAIL={RAIL}\nN0={N0}\nLAM={LAM}\nEV_FLOOR={best_th}\n")
    for j, (ci, code, desc, sg, pr) in enumerate(SPEC):
        fh.write(f"{code}\t{W[j]:.4f}\n")
