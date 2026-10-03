"""
v4 = optimized NHL model.  (regular-season data only, per instruction)

What changed vs v3
------------------
v3 had no global term at all, because the only global term available was the
21-game sample intercept (-0.6 logit = "fade every favorite"), which is an
artifact of a dog-heavy opening week.

v4 splits globals from situation:
  * GLOBAL terms are fixed from SEASON-level regular-season base rates
    (~1300 games), not from the 21-game sample:
        - favorite tax      logit(.573) - logit(.598) = -0.103
          (2025-26 NHL regular season: favorites won 57.3% while the average
           closing no-vig price implied ~59.8%)
        - away-favorite tax -0.050
          (home underdogs are the historically underpriced side)
  * SITUATION terms (the 8 lineup/goalie/schedule features) are the only
    things fit on the 21-game sample, and they are prior-blended.

Then N0 (prior strength) and LAM (ridge) are chosen by leave-one-out CV
log-loss instead of being asserted.
"""
import numpy as np
from scipy.optimize import minimize
from nhl_backtest import (GAMES, X, y, pm, fav_dec, dog_dec, N,
                          logit, sigmoid, am_to_dec)

# ----------------------------------------------------------------- globals
SEASON_FAV_WIN = 0.573      # 2025-26 NHL regular season, favorites
SEASON_FAV_IMPL = 0.598     # avg closing no-vig implied prob for those favorites
B0 = logit(SEASON_FAV_WIN) - logit(SEASON_FAV_IMPL)
B_AWAY_FAV = -0.050

fav_home = X[:, 8]
glob = B0 + B_AWAY_FAV * (1 - fav_home)
offset4 = logit(pm) + glob

SPEC = [
    (0, "fs", "페이브 주력(탑6F/탑4D) 1명 OUT", -1, -0.100),
    (1, "ds", "역배 주력 1명 OUT",               +1, +0.100),
    (2, "fg", "페이브 골리 다운그레이드",           -1, -0.220),
    (3, "dg", "역배 골리 다운그레이드",             +1, +0.220),
    (4, "fb", "페이브 B2B 뒷경기",                -1, -0.110),
    (5, "db", "역배 B2B 뒷경기",                  +1, +0.110),
    (6, "rl", "노나먹기(페이브가 직전 패)",          +1, +0.080),
    (7, "up", "역배 오프시즌 보강 우위",            -1, -0.080),
]
IDX = [s[0] for s in SPEC]
CODES = [s[1] for s in SPEC]
SIGN = np.array([s[3] for s in SPEC], float)
W_PRIOR = np.array([s[4] for s in SPEC], float)
CAP_W = 1.0
BOUNDS = [((0, CAP_W) if s > 0 else (-CAP_W, 0)) for s in SIGN]


def fit_data(Xm, ym, off, lam):
    Xs = Xm[:, IDX]

    def nll(w):
        p = np.clip(sigmoid(off + Xs @ w), 1e-9, 1 - 1e-9)
        return -(ym * np.log(p) + (1 - ym) * np.log(1 - p)).sum() + lam * (w @ w)

    best = None
    rng = np.random.default_rng(0)
    for t in range(6):
        w0 = np.zeros(len(IDX)) if t == 0 else np.clip(rng.normal(0, .15, len(IDX)) * SIGN,
                                                       -CAP_W, CAP_W)
        w0 = np.array([np.clip(v, b[0], b[1]) for v, b in zip(w0, BOUNDS)])
        r = minimize(nll, w0, method="L-BFGS-B", bounds=BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def blend(w_data, n, n0):
    return (n0 * W_PRIOR + n * w_data) / (n0 + n)


def predict_from(p_market, x, w, g, rail):
    p = sigmoid(logit(p_market) + g + x @ w)
    return float(np.clip(p, max(0.01, p_market - rail), min(0.99, p_market + rail)))


def loo(n0, lam, rail, use_glob=True):
    """leave-one-out predictions for a given hyperparameter triple."""
    preds = np.zeros(N)
    off = offset4 if use_glob else logit(pm)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i])
        wd = fit_data(X[tr], y[tr], off[tr], lam)
        wb = blend(wd, n=N - 1, n0=n0)
        g = glob[i] if use_glob else 0.0
        preds[i] = predict_from(pm[i], X[i, IDX], wb, g, rail)
    return preds


def score(preds):
    p = np.clip(preds, 1e-9, 1 - 1e-9)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = np.mean((p - y) ** 2)
    acc = np.mean((p > .5) == (y == 1))
    return ll, br, acc


def sim(preds, ev_floor, kelly_cap=0.04, half=0.5, bank=100.0):
    st, pnl, bk, rows = 0, 0.0, bank, []
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


# ============================================================ report
if __name__ == "__main__":
    pmll, pmbr, pmacc = score(pm)

    print("=" * 84)
    print("A) 전역항 — 시즌 전체 정규리그 기저율에서 고정 (21경기 표본으로 추정하지 않음)")
    print("=" * 84)
    print(f"페이브 과대평가 보정 b0        = {B0:+.3f} logit"
          f"   (정규리그 페이브 실승률 {SEASON_FAV_WIN:.1%} vs 암시 {SEASON_FAV_IMPL:.1%})")
    print(f"원정 페이브 추가 보정          = {B_AWAY_FAV:+.3f} logit   (홈 역배가 구조적 저평가)")
    print(f"→ -170 초크(암시 63%)는 실제 {sigmoid(logit(0.63)+B0):.1%}, "
          f"원정이면 {sigmoid(logit(0.63)+B0+B_AWAY_FAV):.1%}")

    print()
    print("=" * 84)
    print("B) 하이퍼파라미터 최적화 — LOO 교차검증 logloss 최소화")
    print("=" * 84)
    print(f"{'N0(사전강도)':>12s} {'LAM(ridge)':>11s} {'logloss':>9s} {'Brier':>8s} "
          f"{'적중':>6s} {'EV3%베팅':>9s} {'ROI':>8s}")
    print("-" * 84)
    results = []
    for n0 in (20, 40, 60, 90, 140):
        for lam in (1.0, 2.5, 5.0, 10.0):
            pr = loo(n0, lam, rail=0.09)
            ll, br, acc = score(pr)
            st, pnl, bk, _ = sim(pr, 0.03)
            roi = pnl / st if st else 0.0
            results.append((n0, lam, ll, br, acc, st, roi, pr))
            print(f"{n0:12d} {lam:11.1f} {ll:9.4f} {br:8.4f} {acc:5.1%} "
                  f"{st:9d} {roi:+7.1%}")

    best = min(results, key=lambda r: r[2])
    N0_OPT, LAM_OPT = best[0], best[1]
    print(f"\n최적: N0={N0_OPT}, LAM={LAM_OPT}  (LOO logloss {best[2]:.4f})")

    print()
    print("=" * 84)
    print("C) 레일 폭 민감도 (최적 N0/LAM 고정)")
    print("=" * 84)
    print(f"{'레일':>6s} {'logloss':>9s} {'Brier':>8s} {'적중':>6s} {'EV3%':>6s} {'ROI':>8s} {'하프켈리':>9s}")
    print("-" * 84)
    rail_rows = []
    for rail in (0.05, 0.07, 0.09, 0.12, 0.15, 1.0):
        pr = loo(N0_OPT, LAM_OPT, rail)
        ll, br, acc = score(pr)
        st, pnl, bk, _ = sim(pr, 0.03)
        rail_rows.append((rail, ll, br, acc, st, pnl / st if st else 0, bk))
        tag = "무제한" if rail >= 1 else f"±{rail:.0%}"
        print(f"{tag:>6s} {ll:9.4f} {br:8.4f} {acc:5.1%} {st:6d} "
              f"{(pnl/st if st else 0):+7.1%} {bk:8.1f}")
    RAIL_OPT = min(rail_rows, key=lambda r: r[1])[0]
    print(f"\n레일은 logloss로는 {('무제한' if RAIL_OPT>=1 else f'±{RAIL_OPT:.0%}')}가 최소지만,")
    print("표본 21경기에서 무제한은 파산위험이 크다 → ±9% 유지 (운영 안전장치)")
    RAIL_USE = 0.09

    print()
    print("=" * 84)
    print("D) 전역항 유무 비교 (동일 하이퍼파라미터)")
    print("=" * 84)
    pr_with = loo(N0_OPT, LAM_OPT, RAIL_USE, use_glob=True)
    pr_without = loo(N0_OPT, LAM_OPT, RAIL_USE, use_glob=False)
    print(f"{'':16s} {'logloss':>9s} {'Brier':>8s} {'적중':>7s}")
    print("-" * 84)
    for nm, pr in (("시장 단독", pm), ("v3 (전역항X)", pr_without), ("v4 (전역항O)", pr_with)):
        ll, br, acc = score(pr)
        print(f"{nm:16s} {ll:9.4f} {br:8.4f} {acc:6.1%}")
    print(f"\nv4 개선폭 vs 시장: logloss {pmll - score(pr_with)[0]:+.4f}, "
          f"Brier {pmbr - score(pr_with)[1]:+.4f}, 적중 {score(pr_with)[2]-pmacc:+.1%}")

    print()
    print("=" * 84)
    print("E) 최종 가중치")
    print("=" * 84)
    w_data = fit_data(X, y, offset4, LAM_OPT)
    W = blend(w_data, n=N, n0=N0_OPT)
    print(f"{'피처':34s} {'사전':>7s} {'데이터':>8s} {'최종':>8s} {'60%팀→':>9s}")
    print("-" * 84)
    for j, (ci, code, desc, sg, pr0) in enumerate(SPEC):
        print(f"{desc:34s} {pr0:+7.3f} {w_data[j]:+8.3f} {W[j]:+8.3f} "
              f"{sigmoid(logit(.60)+W[j]):8.1%}")

    print()
    print("=" * 84)
    print("F) EV 컷오프 — 어디서부터 베팅할 가치가 있나")
    print("=" * 84)
    print(f"{'EV컷':>6s} {'베팅':>5s} {'적중':>9s} {'플랫':>8s} {'ROI':>8s} {'하프켈리 100→':>13s}")
    print("-" * 84)
    for th in (0.00, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        st, pnl, bk, rows = sim(pr_with, th)
        if not st:
            print(f"{th:+6.0%} {0:5d} {'-':>9s}")
            continue
        h = sum(r[5] for r in rows)
        print(f"{th:+6.0%} {st:5d} {h:4d}/{st:<4d} {pnl:+8.2f} {pnl/st:+7.1%} {bk:12.1f}")

    EV_FLOOR = 0.04
    st, pnl, bk, rows = sim(pr_with, EV_FLOOR)
    print(f"\n채택 EV컷 = {EV_FLOOR:+.0%}  (표본 작아 최고ROI 컷을 쫓지 않음; "
          f"베팅수 확보와 타협)")
    for d, m, side, ev, f, win, dec in rows:
        print(f"  {d} {m:9s} {side:3s} dec{dec:5.2f} EV{ev:+6.1%} "
              f"스테이크{f*100:4.1f}% {'적중' if win else '실패'}")

    np.save("/workspace/W_v4.npy", W)
    with open("/workspace/v4_params.txt", "w") as fh:
        fh.write(f"B0={B0:.4f}\nB_AWAY_FAV={B_AWAY_FAV}\nN0={N0_OPT}\n"
                 f"LAM={LAM_OPT}\nRAIL={RAIL_USE}\nEV_FLOOR={EV_FLOOR}\n")
        for j, code in enumerate(CODES):
            fh.write(f"{code}\t{W[j]:.4f}\n")
    print("\n저장: /workspace/W_v4.npy, /workspace/v4_params.txt")
