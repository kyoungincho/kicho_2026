"""
NHL backtest v2 -- sign-constrained ridge on top of the market price.

Why v2:
  v1's unconstrained fit put a big negative weight on "favorite is home".
  17 of 21 favorites were home and the whole opening week was a dog week
  (favorites 42.9% vs 57.7% implied), so that term was absorbing a global
  small-sample drift, not a repeatable effect. Long-run NHL favorites win
  ~57%, so we do NOT let the model learn "fade every favorite".

v2 rules:
  - drop the home term (confounded + already in the price)
  - no free intercept (optional, reported separately)
  - sign constraints from domain logic, so 21 games cannot flip a sign
  - ridge penalty tuned by leave-one-out CV
  - final weights shrunk toward 0 by k = n/(n+n0)
"""
import numpy as np
from scipy.optimize import minimize
from nhl_backtest import (GAMES, FEAT, X, y, pm, fav_dec, dog_dec, offset, N,
                          logit, sigmoid, am_to_dec)

# feature index -> (name, sign)  sign=-1 means weight must be <=0
SPEC = [
    (0, "fs", "페이브 주력 1명 OUT",        -1),
    (1, "ds", "역배 주력 1명 OUT",          +1),
    (2, "fg", "페이브 골리 다운그레이드",      -1),
    (3, "dg", "역배 골리 다운그레이드",        +1),
    (4, "fb", "페이브 B2B 뒷경기",           -1),
    (5, "db", "역배 B2B 뒷경기",             +1),
    (6, "rl", "노나먹기(직전 페이브 패)",      +1),
    (7, "up", "역배 오프시즌 보강 우위",       -1),
]
IDX = [s[0] for s in SPEC]
SIGN = np.array([s[3] for s in SPEC], dtype=float)
CAP = 1.2  # max |w| per feature, keeps one feature from dominating


def fit_constrained(Xm, ym, off, lam, use_intercept=False):
    Xs = Xm[:, IDX]
    k = Xs.shape[1]
    nparam = k + (1 if use_intercept else 0)
    # parametrise w = sign * u^2 is non-smooth; use bounds instead
    bounds = []
    for s in SIGN:
        bounds.append((0, CAP) if s > 0 else (-CAP, 0))
    if use_intercept:
        bounds.append((-0.6, 0.6))

    def nll(theta):
        w = theta[:k]
        b = theta[k] if use_intercept else 0.0
        z = off + Xs @ w + b
        p = np.clip(sigmoid(z), 1e-9, 1 - 1e-9)
        pen = lam * (w @ w) + (lam * b * b if use_intercept else 0.0)
        return -(ym * np.log(p) + (1 - ym) * np.log(1 - p)).sum() + pen

    best = None
    rng = np.random.default_rng(0)
    for t in range(6):
        th0 = np.zeros(nparam) if t == 0 else np.clip(
            rng.normal(0, .2, nparam) * np.concatenate(
                [SIGN, [1]] if use_intercept else [SIGN]), -CAP, CAP)
        th0 = np.array([np.clip(v, b[0], b[1]) for v, b in zip(th0, bounds)])
        r = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    w = best.x[:k]
    b = best.x[k] if use_intercept else 0.0
    return w, b


def loo(lam, use_intercept=False, shrink=1.0):
    ll = br = 0.0
    acc = 0
    preds = np.zeros(N)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i])
        w, b = fit_constrained(X[tr], y[tr], offset[tr], lam, use_intercept)
        p = float(sigmoid(offset[i] + shrink * (X[i, IDX] @ w + b)))
        preds[i] = p
        ll -= y[i] * np.log(max(p, 1e-9)) + (1 - y[i]) * np.log(max(1 - p, 1e-9))
        br += (p - y[i]) ** 2
        acc += int((p > .5) == (y[i] == 1))
    return ll / N, br / N, acc / N, preds


def market_loo():
    ll = br = 0.0
    acc = 0
    for i in range(N):
        q = pm[i]
        ll -= y[i] * np.log(q) + (1 - y[i]) * np.log(1 - q)
        br += (q - y[i]) ** 2
        acc += int((q > .5) == (y[i] == 1))
    return ll / N, br / N, acc / N


def bet_sim(preds, thresh, kelly_cap=0.05, bankroll=100.0):
    """flat-1u ROI plus fractional-Kelly bankroll growth."""
    st = 0
    pnl = 0.0
    bk = bankroll
    rows = []
    for i in range(N):
        ef = preds[i] * fav_dec[i] - 1
        ed = (1 - preds[i]) * dog_dec[i] - 1
        if max(ef, ed) < thresh:
            continue
        if ef >= ed:
            p, dec, win, side, ev = preds[i], fav_dec[i], y[i] == 1, "FAV", ef
        else:
            p, dec, win, side, ev = 1 - preds[i], dog_dec[i], y[i] == 0, "DOG", ed
        b = dec - 1
        f = max(0.0, min(kelly_cap, (p * b - (1 - p)) / b))
        stake_frac = f * 0.5  # half-Kelly
        bk += bk * stake_frac * (b if win else -1)
        st += 1
        pnl += b if win else -1
        rows.append((GAMES[i][0], GAMES[i][1], side, ev, stake_frac, win))
    return st, pnl, bk, rows


print("=" * 78)
print("A) 부호제약 ridge — lambda 튜닝 (LOO)")
print("=" * 78)
mll, mbr, macc = market_loo()
print(f"시장 단독:  logloss {mll:.4f}   Brier {mbr:.4f}   적중 {macc:.1%}\n")
print(f"{'lam':>6s} {'절편':>5s} {'logloss':>9s} {'개선':>8s} {'Brier':>8s} {'적중':>6s} "
      f"{'베팅(EV5%)':>10s} {'ROI':>8s}")
print("-" * 78)
results = []
for use_b in (False, True):
    for lam in (0.5, 1, 2, 3, 5, 8, 12, 20):
        ll, br, acc, preds = loo(lam, use_b)
        st, pnl, bk, _ = bet_sim(preds, 0.05)
        results.append((lam, use_b, ll, br, acc, st, pnl / st if st else 0, preds))
        print(f"{lam:6.1f} {'Y' if use_b else 'N':>5s} {ll:9.4f} {mll-ll:+8.4f} "
              f"{br:8.4f} {acc:5.1%} {st:10d} {(pnl/st if st else 0):+7.1%}")

best = min(results, key=lambda r: r[2])
lam_b, useb_b = best[0], best[1]
print(f"\n최적: lam={lam_b}, 절편={'사용' if useb_b else '미사용'}  "
      f"-> logloss {best[2]:.4f} (시장 대비 {mll-best[2]:+.4f})")

print()
print("=" * 78)
print("B) 최종 가중치 (전체 21경기 적합) + 축소")
print("=" * 78)
w_full, b_full = fit_constrained(X, y, offset, lam_b, useb_b)
n0 = 60.0
k_shrink = N / (N + n0)
print(f"표본 n={N}, 사전표본 n0={n0:.0f}  ->  축소계수 k={k_shrink:.3f}\n")
print(f"{'피처':30s} {'원시 w':>9s} {'축소 w':>9s} {'60%팀 승률 영향':>20s}")
print("-" * 78)
rows_out = []
for j, (ci, code, desc, sg) in enumerate(SPEC):
    wr = w_full[j]
    ws = wr * k_shrink
    p1 = sigmoid(logit(0.60) + ws)
    print(f"{desc:30s} {wr:+9.3f} {ws:+9.3f} {f'60.0% -> {p1*100:.1f}%':>20s}")
    rows_out.append((code, desc, wr, ws))
if useb_b:
    print(f"{'(절편)':30s} {b_full:+9.3f} {b_full*k_shrink:+9.3f}")

ll, br, acc, preds = loo(lam_b, useb_b, shrink=k_shrink)
print(f"\n축소 적용 LOO:  logloss {ll:.4f} (시장 {mll:.4f})  Brier {br:.4f}  적중 {acc:.1%}")

print()
print("=" * 78)
print("C) EV 컷오프별 성적 (축소 적용, LOO 예측)")
print("=" * 78)
print(f"{'EV컷':>6s} {'베팅':>5s} {'적중':>6s} {'수익(1u)':>10s} {'ROI':>8s} {'하프켈리 뱅크롤':>16s}")
print("-" * 78)
for th in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
    st, pnl, bk, rows = bet_sim(preds, th)
    if st == 0:
        print(f"{th:+6.0%} {0:5d}      -          -        -                -")
        continue
    hits = sum(r[5] for r in rows)
    print(f"{th:+6.0%} {st:5d} {hits:3d}/{st:<3d} {pnl:+10.2f} {pnl/st:+7.1%} "
          f"{bk:15.1f}")

st, pnl, bk, rows = bet_sim(preds, 0.04)
print(f"\nEV>=4% 내역:")
for d, m, side, ev, f, win in rows:
    print(f"  {d} {m:9s} {side:3s} EV{ev:+.3f} 스테이크{f*100:4.1f}%  {'적중' if win else '실패'}")

np.save("/workspace/nhl_w_v2.npy", w_full * k_shrink)
with open("/workspace/nhl_model_v2.txt", "w") as f:
    f.write(f"lam={lam_b}\nintercept={useb_b}:{b_full}\nk={k_shrink}\n")
    for code, desc, wr, ws in rows_out:
        f.write(f"{code}\t{desc}\t{wr:.4f}\t{ws:.4f}\n")
print("\n저장: /workspace/nhl_w_v2.npy, nhl_model_v2.txt")
