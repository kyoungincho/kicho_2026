"""
Soccer (UEFA Nations League) optimized model.

Structural difference vs hockey: 1X2 has three outcomes, so a favorite can
fail without losing. The backtest says that is where all the money is:

    tier gap present : 20 matches, 90.0% win,  5.0% draw   (implied 59.6%)
    tier gap absent  : 16 matches, 25.0% win, 43.8% draw   (implied 48.0%)

So the single most valuable soccer parameter is not price, it is whether a
real quality gap exists. Without one, the favorite's price is a trap and the
DRAW is the mispriced side.

Model:
    logit(P(fav wins)) = logit(p_market_novig) + w_tier*tier + w_notier*(1-tier)
                                               + w_away*(1-home)
plus a separate draw estimate for the no-gap bucket.

Caveat kept in the output: odds here are reconstructed closing prices, and the
tier_gap label was assigned by me, so it carries hindsight risk.
"""
import numpy as np
from scipy.optimize import minimize
from soccer_backtest import M, res, dec, tier, home, win

OVERROUND = 1.06          # typical 1X2 book margin -> de-vig the favorite price
N = len(M)
p_mkt = np.clip((1.0 / dec) / OVERROUND, 0.02, 0.95)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# draw price as a function of the favorite's price (structural approximation of
# the 1X2 market, not per-match quotes)
_DP_X = np.array([1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70])
_DP_Y = np.array([6.00, 4.75, 4.00, 3.60, 3.40, 3.25, 3.25, 3.35])
draw_dec = np.interp(dec, _DP_X, _DP_Y)
is_draw = (res == "D").astype(float)
is_loss = (res == "L").astype(float)
# back out the third price from the book: 1/fav + 1/draw + 1/dog = OVERROUND
_q_dog = np.clip(OVERROUND - 1 / dec - 1 / draw_dec, 0.02, 0.9)
dog_dec = 1 / _q_dog

# Priors are deliberately NEAR-ZERO for the tier term, because the market
# already prices quality gaps and international football markets are efficient.
# The no-gap term gets a real negative prior: overpricing home advantage and
# narrative in evenly-matched internationals (and the matching draw underpricing)
# is a documented bias, not something I discovered in 36 matches.
SPEC = [
    ("tier",   "티어갭 있음",        +1, +0.10),
    ("notier", "티어갭 없음",        -1, -0.25),
    ("away",   "원정 페이브",        +1, +0.05),
]
SIGN = np.array([s[2] for s in SPEC], float)
W_PRIOR = np.array([s[3] for s in SPEC], float)
CAP = 1.2
BOUNDS = [((0, CAP) if s > 0 else (-CAP, 0)) for s in SIGN]

Z = np.column_stack([tier, 1 - tier, 1 - home]).astype(float)
off = logit(p_mkt)


def fit(Zm, ym, offm, lam):
    def nll(w):
        p = np.clip(sigmoid(offm + Zm @ w), 1e-9, 1 - 1e-9)
        return -(ym * np.log(p) + (1 - ym) * np.log(1 - p)).sum() + lam * (w @ w)

    best = None
    rng = np.random.default_rng(0)
    for t in range(6):
        w0 = np.zeros(Zm.shape[1]) if t == 0 else np.clip(
            rng.normal(0, .2, Zm.shape[1]) * SIGN, -CAP, CAP)
        w0 = np.array([np.clip(v, b[0], b[1]) for v, b in zip(w0, BOUNDS)])
        r = minimize(nll, w0, method="L-BFGS-B", bounds=BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def blend(wd, n, n0):
    return (n0 * W_PRIOR + n * wd) / (n0 + n)


def loo(n0, lam, rail):
    pr = np.zeros(N)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i])
        wd = fit(Z[tr], win[tr], off[tr], lam)
        wb = blend(wd, n=N - 1, n0=n0)
        p = sigmoid(off[i] + Z[i] @ wb)
        pr[i] = float(np.clip(p, max(.01, p_mkt[i] - rail), min(.97, p_mkt[i] + rail)))
    return pr


def score(pr):
    p = np.clip(pr, 1e-9, 1 - 1e-9)
    ll = -np.mean(win * np.log(p) + (1 - win) * np.log(1 - p))
    br = np.mean((p - win) ** 2)
    acc = np.mean((p > .5) == (win == 1))
    return ll, br, acc


def fav_roi(pr, floor):
    st, h, pnl = 0, 0, 0.0
    for i in range(N):
        ev = pr[i] * dec[i] - 1
        if ev < floor:
            continue
        st += 1
        w = win[i] == 1
        h += int(w)
        pnl += (dec[i] - 1) if w else -1
    return st, h, pnl, (pnl / st if st else 0.0)


print("=" * 84)
print("K) 축구 — 하이퍼파라미터 최적화 (LOO, 36경기)")
print("=" * 84)
mll, mbr, macc = score(p_mkt)
print(f"{'N0':>5s} {'LAM':>6s} {'레일':>6s} {'logloss':>9s} {'Brier':>8s} {'적중':>7s}")
print("-" * 84)
rows = []
for n0 in (10, 20, 35, 60, 120):
    for lam in (1.0, 3.0, 8.0):
        for rail in (0.12, 0.20, 0.30):
            pr = loo(n0, lam, rail)
            ll, br, acc = score(pr)
            rows.append((n0, lam, rail, ll, br, acc, pr))
            print(f"{n0:5d} {lam:6.1f} {rail:5.0%} {ll:9.4f} {br:8.4f} {acc:6.1%}")
best = min(rows, key=lambda r: r[3])
print(f"\nLOO 최소: N0={best[0]}, LAM={best[1]}, 레일 ±{best[2]:.0%}  "
      f"(logloss {best[3]:.4f}, 적중 {best[5]:.1%})")
print(f"시장 단독 : logloss {mll:.4f}  Brier {mbr:.4f}  적중 {macc:.1%}")

# ---- hindsight-bias guard -------------------------------------------------
# LOO cannot detect label leakage: I assigned tier_gap AFTER seeing results, so
# a free-running fit will always look brilliant. Hold N0 high on purpose.
N0_S, LAM_S, RAIL_S = 60, 3.0, 0.20
pr_s = loo(N0_S, LAM_S, RAIL_S)
ll_s, br_s, acc_s = score(pr_s)
print(f"\n채택: N0={N0_S}, LAM={LAM_S}, 레일 ±{RAIL_S:.0%}  "
      f"(logloss {ll_s:.4f}, Brier {br_s:.4f}, 적중 {acc_s:.1%})")
print("LOO가 고른 N0=10을 안 쓰는 이유: 티어갭 라벨을 결과를 본 뒤 내가 붙였다.")
print("LOO는 라벨 누수를 잡아내지 못하므로 사전값을 강제로 무겁게 둔다.")

# ---- temporal holdout ----------------------------------------------------
print()
print("시간 분할 검증 (앞 24경기 학습 -> 뒤 12경기 예측):")
tr = np.arange(24)
te = np.arange(24, N)
wd_h = fit(Z[tr], win[tr], off[tr], LAM_S)
for n0h in (10, 60):
    wb_h = blend(wd_h, n=len(tr), n0=n0h)
    ph = np.clip(sigmoid(off[te] + Z[te] @ wb_h),
                 np.maximum(.01, p_mkt[te] - RAIL_S), np.minimum(.97, p_mkt[te] + RAIL_S))
    llh = -np.mean(win[te] * np.log(ph) + (1 - win[te]) * np.log(1 - ph))
    llm = -np.mean(win[te] * np.log(p_mkt[te]) + (1 - win[te]) * np.log(1 - p_mkt[te]))
    acch = np.mean((ph > .5) == (win[te] == 1))
    print(f"  N0={n0h:3d}: logloss {llh:.4f} (시장 {llm:.4f}, 개선 {llm-llh:+.4f})  "
          f"적중 {acch:.1%}")

print()
print("=" * 84)
print("L) 최종 가중치")
print("=" * 84)
wd = fit(Z, win, off, LAM_S)
W = blend(wd, n=N, n0=N0_S)
print(f"{'항':20s} {'사전':>7s} {'데이터':>8s} {'최종':>8s} {'단배1.80(55%)→':>15s}")
print("-" * 84)
for j, (code, desc, sg, pr0) in enumerate(SPEC):
    print(f"{desc:20s} {pr0:+7.3f} {wd[j]:+8.3f} {W[j]:+8.3f} "
          f"{sigmoid(logit(.55)+W[j]):14.1%}")

print()
print("=" * 84)
print("M) 페이브 ML EV컷별 성적")
print("=" * 84)
print(f"{'EV컷':>6s} {'베팅':>5s} {'적중':>9s} {'수익':>8s} {'ROI':>8s}")
print("-" * 84)
for th in (0.00, 0.05, 0.10, 0.15, 0.25):
    st, h, pnl, roi = fav_roi(pr_s, th)
    if not st:
        print(f"{th:+6.0%} {0:5d}")
        continue
    print(f"{th:+6.0%} {st:5d} {h:4d}/{st:<4d} {pnl:+8.2f} {roi:+7.1%}")

print()
print("=" * 84)
print("N) 무승부 — 티어갭 없는 경기의 진짜 가치")
print("=" * 84)
ntg = tier == 0
tg = tier == 1
print(f"{'버킷':24s} {'N':>3s} {'무승부율':>8s} {'무승부 암시':>10s} {'무승부 ROI':>11s}")
print("-" * 84)
for nm, m in (("티어갭 없음", ntg), ("티어갭 없음 & 홈페이브", ntg & (home == 1)),
              ("티어갭 있음", tg), ("전체", np.ones(N, bool))):
    n = m.sum()
    dr = is_draw[m].mean()
    imp = (1 / draw_dec[m]).mean()
    roi = np.where(is_draw[m] == 1, draw_dec[m] - 1, -1.0).mean()
    print(f"{nm:24s} {n:3d} {dr:7.1%} {imp:9.1%} {roi:+10.1%}")

print("\n주의: 티어갭 없음의 43.8% 무승부는 사실상 '홈페이브가 아닌 4경기'가 전부다")
print("      (그 4경기 전부 무승부). 홈페이브 12경기는 25%로 암시보다 낮다.")
print("      -> 무승부 자체를 edge로 쓰기엔 표본이 4경기. 단독 베팅 근거로 부족.")

# shrunk draw estimate for the no-gap bucket
emp_d = is_draw[ntg].mean()
imp_d = (1 / draw_dec[ntg]).mean()
n_d = int(ntg.sum())
N0_D = 80.0           # heavy prior: the raw number rests on 4 matches
p_draw_ngap = (N0_D * imp_d + n_d * emp_d) / (N0_D + n_d)
print(f"\n티어갭 없는 경기 무승부 확률:")
print(f"  경험값 {emp_d:.1%} (n={n_d}) / 시장암시 {imp_d:.1%} / 강하게 축소 후 {p_draw_ngap:.1%}")
print(f"  무승부 단배 3.40 기준 EV = {p_draw_ngap*3.40-1:+.1%}")

print()
print("=" * 84)
print("N2) 그럼 티어갭 없는 경기에서 뭘 사야 하나 — 역배(언더독 승)")
print("=" * 84)
print(f"{'버킷':24s} {'N':>3s} {'역배승률':>8s} {'역배 평균배당':>12s} {'역배 ROI':>9s}")
print("-" * 84)
for nm, m in (("티어갭 없음", ntg), ("티어갭 없음 & 홈페이브", ntg & (home == 1)),
              ("티어갭 있음", tg)):
    n = m.sum()
    lr = is_loss[m].mean()
    roi = np.where(is_loss[m] == 1, dog_dec[m] - 1, -1.0).mean()
    print(f"{nm:24s} {n:3d} {lr:7.1%} {dog_dec[m].mean():11.2f} {roi:+8.1%}")
print("\n-> 티어갭 없는 경기는 '페이브를 사지 않는다'가 1차 결론.")
print("   역배 쪽도 플러스지만 표본이 작다. 가장 안전한 행동은 PASS.")

print()
print("=" * 84)
print("O) 축구 파라미터 확정")
print("=" * 84)
print(f"S2 티어갭 있음        w = {W[0]:+.3f} logit  (사전 {W_PRIOR[0]:+.3f})")
print(f"S3 티어갭 없음        w = {W[1]:+.3f} logit  -> 홈빨/서사만으로는 페이브 금지")
print(f"S7 원정 페이브        w = {W[2]:+.3f} logit  (원정 페이브 할인은 과도)")
print(f"S8 무승부(티어갭X)    p = {p_draw_ngap:.1%}  -> 단배 3.0+ 에서 소액만. 축 금지.")
print(f"레일 ±{RAIL_S:.0%} / EV컷 +10% (1X2는 분산이 커 하키보다 컷을 높게)")
print()
print("주의: 배당은 복원된 종가 근사치이고 티어갭 라벨은 내가 부여했다.")
print("      방향은 신뢰할 수 있으나 크기는 과대추정 가능성 있음.")

np.save("/workspace/W_soccer_v4.npy", W)
with open("/workspace/soccer_v4_params.txt", "w") as fh:
    fh.write(f"N0={N0_S}\nLAM={LAM_S}\nRAIL={RAIL_S}\nEV_FLOOR=0.10\n")
    fh.write(f"P_DRAW_NOGAP={p_draw_ngap:.4f}\n")
    for j, (code, *_ ) in enumerate(SPEC):
        fh.write(f"{code}\t{W[j]:.4f}\n")
print("\n저장: /workspace/W_soccer_v4.npy, /workspace/soccer_v4_params.txt")
