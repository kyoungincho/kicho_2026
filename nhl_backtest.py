"""
NHL 2026-27 regular season backtest (Sep 29 - Oct 2, 2026) -- 21 games.
Goal: find whether situational features add information ON TOP OF the closing market price,
then turn that into calibrated win probabilities and bet EV.

Model: logit(p_fav) = logit(p_market_novig) + sum_i w_i * x_i
  -> w learned with L2 ridge; market price is an offset, so w measures *edge over the market*.
Validation: leave-one-out CV log-loss + Brier vs market-only baseline.
"""
import numpy as np
from itertools import product
from scipy.optimize import minimize

# ---------------------------------------------------------------- data
# Each row = one game, oriented from the MARKET FAVORITE's perspective.
# fav_ml / dog_ml = closing American odds. fav_win = 1 if favorite won (incl. OT/SO).
# features (favorite's perspective):
#   fs = favorite key skaters out (top-6 F / top-4 D count)
#   ds = dog key skaters out
#   fg = favorite goalie downgrade (backup/suspended/collapsing tandem) 0/1
#   dg = dog goalie downgrade 0/1
#   fb = favorite on 2nd of back-to-back 0/1
#   db = dog on 2nd of back-to-back 0/1
#   rl = favorite LOST the previous meeting this season (노나먹기 -> favors favorite) 0/1
#   up = dog has clear offseason roster upgrade vs favorite 0/1
#   fh = favorite is home 0/1
GAMES = [
    # date, matchup, fav, fav_ml, dog_ml, fav_win, fs, ds, fg, dg, fb, db, rl, up, fh
    ("09-29", "FLA@CAR", "CAR", -127, +108, 0,  1, 0, 1, 0, 0, 0, 0, 0, 1),
    ("09-29", "MTL@TOR", "TOR", -115, -105, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1),
    ("09-29", "NYR@BOS", "BOS", -113, -103, 1,  1, 0, 0, 0, 0, 0, 0, 0, 1),
    ("09-29", "VAN@EDM", "EDM", -270, +230, 0,  1, 0, 1, 0, 0, 0, 0, 0, 1),
    ("09-29", "CHI@VGK", "VGK", -250, +210, 1,  0, 1, 0, 0, 0, 0, 0, 0, 1),
    ("09-30", "PIT@PHI", "PHI", -144, +120, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1),
    ("09-30", "NYI@TOR", "TOR", -125, +104, 1,  0, 1, 0, 0, 1, 0, 0, 0, 1),
    ("09-30", "LAK@COL", "COL", -192, +160, 1,  0, 1, 0, 0, 0, 0, 0, 0, 1),
    ("10-01", "TB@NYR",  "TB",  -138, +116, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("10-01", "PHI@NJD", "NJD", -140, +120, 1,  0, 0, 0, 0, 0, 1, 0, 0, 1),
    ("10-01", "BUF@CBJ", "CBJ", -110, -110, 1,  0, 0, 0, 0, 0, 0, 0, 0, 1),
    ("10-01", "MIN@NSH", "MIN", -140, +120, 1,  0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("10-01", "SEA@CGY", "CGY", -115, -105, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1),
    ("10-01", "CHI@UTA", "UTA", -210, +170, 1,  0, 2, 0, 0, 0, 0, 0, 0, 1),
    ("10-01", "EDM@VAN", "EDM", -200, +160, 1,  2, 0, 1, 0, 0, 0, 1, 0, 0),
    ("10-01", "FLA@SJS", "FLA", -135, +115, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("10-02", "NYR@DET", "DET", -132, +110, 0,  2, 0, 0, 1, 0, 1, 0, 0, 1),
    ("10-02", "WSH@CAR", "CAR", -150, +125, 0,  1, 0, 1, 0, 0, 1, 0, 1, 1),
    ("10-02", "BOS@WPG", "WPG", -125, +105, 0,  0, 1, 1, 0, 0, 0, 0, 0, 1),
    ("10-02", "STL@DAL", "DAL", -183, +158, 0,  1, 0, 0, 0, 0, 0, 0, 0, 1),
    ("10-02", "ANA@VGK", "VGK", -190, +168, 0,  0, 1, 0, 0, 0, 0, 0, 0, 1),
]

FEAT = ["fs", "ds", "fg", "dg", "fb", "db", "rl", "up", "fh"]


def am_to_dec(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def am_to_prob(a):
    d = am_to_dec(a)
    return 1.0 / d


def novig(fav_ml, dog_ml):
    """Remove vig proportionally -> true-ish market prob for the favorite."""
    pf, pd = am_to_prob(fav_ml), am_to_prob(dog_ml)
    return pf / (pf + pd)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# build matrices
X = np.array([[g[6 + i] for i in range(len(FEAT))] for g in GAMES], dtype=float)
y = np.array([g[5] for g in GAMES], dtype=float)
pm = np.array([novig(g[3], g[4]) for g in GAMES])
fav_dec = np.array([am_to_dec(g[3]) for g in GAMES])
dog_dec = np.array([am_to_dec(g[4]) for g in GAMES])
offset = logit(pm)
N = len(GAMES)


# ---------------------------------------------------------------- univariate audit
def univariate_report():
    print("=" * 74)
    print("1) 피처별 단변량 성적 (시장 종가 기준, 21경기)")
    print("=" * 74)
    base_fav = y.mean()
    print(f"기준선: 페이브 승률 {base_fav:.1%}  (시장 평균 암시 {pm.mean():.1%})")
    print(f"  -> 개막주 페이브가 시장 예상보다 {pm.mean()-base_fav:+.1%} 미달 (역배 장세)\n")

    print(f"{'조건':38s} {'N':>3s} {'페이브승':>7s} {'역배ROI':>9s}")
    print("-" * 74)

    def line(name, mask):
        m = mask.astype(bool)
        n = m.sum()
        if n == 0:
            return
        fw = y[m].mean()
        # ROI of blindly backing the DOG in these spots
        ret = np.where(y[m] == 0, dog_dec[m] - 1, -1.0).sum()
        print(f"{name:38s} {n:3d} {fw:6.1%} {ret/n:+8.1%}")

    fs, ds = X[:, 0], X[:, 1]
    fg, dg = X[:, 2], X[:, 3]
    fb, db = X[:, 4], X[:, 5]
    rl, up = X[:, 6], X[:, 7]

    line("전체", np.ones(N))
    line("페이브 주력선수 OUT (fs>=1)", fs >= 1)
    line("  fs>=1 & 역배는 멀쩡(ds=0)", (fs >= 1) & (ds == 0))
    line("페이브 골리 다운그레이드", fg >= 1)
    line("페이브 구멍(선수or골리)", (fs >= 1) | (fg >= 1))
    line("역배 주력선수 OUT (ds>=1)", ds >= 1)
    line("  ds>=1 & 페이브는 멀쩡", (ds >= 1) & (fs == 0) & (fg == 0))
    line("구멍 차이 = 역배쪽이 더 큼", (ds + dg) > (fs + fg))
    line("구멍 차이 = 페이브쪽이 더 큼", (fs + fg) > (ds + dg))
    line("구멍 동일", (fs + fg) == (ds + dg))
    line("초크 ML <= -170", np.array([g[3] <= -170 for g in GAMES]))
    line("  초크 & 페이브 구멍 있음", np.array([g[3] <= -170 for g in GAMES]) & ((fs >= 1) | (fg >= 1)))
    line("  초크 & 역배 구멍 있음", np.array([g[3] <= -170 for g in GAMES]) & (ds >= 1))
    line("페이브 B2B 뒷경기", fb >= 1)
    line("역배 B2B 뒷경기", db >= 1)
    line("노나먹기(페이브가 직전 패)", rl >= 1)
    line("역배 오프시즌 보강 우위", up >= 1)
    line("페이브 홈", X[:, 8] >= 1)
    line("페이브 원정", X[:, 8] == 0)
    print()


# ---------------------------------------------------------------- fit
def fit(Xm, ym, off, lam, idx):
    """ridge logistic with market offset; returns weight vector over idx columns."""
    Xs = Xm[:, idx]
    k = Xs.shape[1]

    def nll(w):
        z = off + Xs @ w
        p = sigmoid(z)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(ym * np.log(p) + (1 - ym) * np.log(1 - p)).sum() + lam * (w @ w)

    best = None
    for seed in range(3):
        w0 = np.zeros(k) if seed == 0 else np.random.default_rng(seed).normal(0, .2, k)
        r = minimize(nll, w0, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def loo_eval(idx, lam):
    """leave-one-out CV: log-loss + brier of model vs market-only."""
    ll_m = ll_k = br_m = br_k = 0.0
    acc_m = acc_k = 0
    preds = np.zeros(N)
    for i in range(N):
        tr = np.array([j for j in range(N) if j != i])
        w = fit(X[tr], y[tr], offset[tr], lam, idx)
        p = float(sigmoid(offset[i] + X[i, idx] @ w))
        preds[i] = p
        q = pm[i]
        ll_k -= y[i] * np.log(max(p, 1e-9)) + (1 - y[i]) * np.log(max(1 - p, 1e-9))
        ll_m -= y[i] * np.log(max(q, 1e-9)) + (1 - y[i]) * np.log(max(1 - q, 1e-9))
        br_k += (p - y[i]) ** 2
        br_m += (q - y[i]) ** 2
        acc_k += int((p > .5) == (y[i] == 1))
        acc_m += int((q > .5) == (y[i] == 1))
    return dict(ll=ll_k / N, ll_mkt=ll_m / N, br=br_k / N, br_mkt=br_m / N,
                acc=acc_k / N, acc_mkt=acc_m / N, preds=preds)


def roi_from_preds(preds, thresh):
    """bet whichever side the model says has edge >= thresh; flat 1u."""
    stake = pnl = 0
    bets = []
    for i in range(N):
        ef = preds[i] * fav_dec[i] - 1          # EV favorite
        ed = (1 - preds[i]) * dog_dec[i] - 1    # EV dog
        if max(ef, ed) < thresh:
            continue
        stake += 1
        if ef >= ed:
            win = y[i] == 1
            pnl += (fav_dec[i] - 1) if win else -1
            bets.append((GAMES[i][1], "FAV", round(ef, 3), win))
        else:
            win = y[i] == 0
            pnl += (dog_dec[i] - 1) if win else -1
            bets.append((GAMES[i][1], "DOG", round(ed, 3), win))
    return stake, pnl, bets


# ---------------------------------------------------------------- feature selection
def select():
    print("=" * 74)
    print("2) 피처셋 선택 — LOO 교차검증 (시장 오프셋 위 증분)")
    print("=" * 74)
    cands = {
        "없음(시장만)": [],
        "fs":                [0],
        "fs,ds":             [0, 1],
        "fs,ds,fg":          [0, 1, 2],
        "fs,ds,fg,dg":       [0, 1, 2, 3],
        "fs,ds,fg,dg,db":    [0, 1, 2, 3, 5],
        "fs,ds,fg,dg,rl":    [0, 1, 2, 3, 6],
        "fs,ds,fg,dg,rl,up": [0, 1, 2, 3, 6, 7],
        "전체 9피처":         list(range(9)),
    }
    rows = []
    for name, idx in cands.items():
        for lam in (0.5, 1.0, 2.0, 4.0, 8.0):
            if not idx:
                r = loo_eval([0], 1e9)  # weights forced ~0 => market only
            else:
                r = loo_eval(idx, lam)
            st, pnl, _ = roi_from_preds(r["preds"], 0.05)
            rows.append((name, lam, r["ll"], r["br"], r["acc"], st, pnl / st if st else 0))
            if not idx:
                break
    print(f"{'피처셋':22s} {'lam':>5s} {'LOO logloss':>12s} {'Brier':>7s} {'적중':>6s} {'베팅':>5s} {'ROI':>8s}")
    print("-" * 74)
    base = None
    for name, lam, ll, br, acc, st, roi in rows:
        if base is None:
            base = ll
        mark = ""
        if ll < base - 1e-9:
            mark = "  <<"
        print(f"{name:22s} {lam:5.1f} {ll:12.4f} {br:7.4f} {acc:5.1%} {st:5d} {roi:+7.1%}{mark}")
    print(f"\n시장 단독 기준: logloss {loo_eval([0],1e9)['ll_mkt']:.4f} / "
          f"Brier {loo_eval([0],1e9)['br_mkt']:.4f} / 적중 {loo_eval([0],1e9)['acc_mkt']:.1%}")
    print()
    best = min(rows[1:], key=lambda r: r[2])
    return cands[best[0]], best[1], best[0]


# ---------------------------------------------------------------- final
def final(idx, lam, name):
    print("=" * 74)
    print(f"3) 최종 파라미터 (피처셋={name}, ridge lam={lam})")
    print("=" * 74)
    w = fit(X, y, offset, lam, idx)
    r = loo_eval(idx, lam)
    print("logit(P(페이브승)) = logit(시장무공제확률) + Σ w·x\n")
    print(f"{'피처':34s} {'w':>8s} {'해석(승률 영향)':>22s}")
    print("-" * 74)
    desc = {
        "fs": "페이브 주력 1명 OUT", "ds": "역배 주력 1명 OUT",
        "fg": "페이브 골리 다운그레이드", "dg": "역배 골리 다운그레이드",
        "fb": "페이브 B2B 뒷경기", "db": "역배 B2B 뒷경기",
        "rl": "노나먹기(직전 페이브 패)", "up": "역배 보강 우위",
        "fh": "페이브 홈",
    }
    for j, c in enumerate(idx):
        # effect at p=0.60 baseline
        p0 = 0.60
        p1 = sigmoid(logit(p0) + w[j])
        print(f"{desc[FEAT[c]]:34s} {w[j]:+8.3f} {f'{p0:.0%} -> {p1:.1%}':>22s}")
    print()
    print(f"LOO 성능:  logloss {r['ll']:.4f} (시장 {r['ll_mkt']:.4f}, "
          f"개선 {r['ll_mkt']-r['ll']:+.4f})")
    print(f"           Brier   {r['br']:.4f} (시장 {r['br_mkt']:.4f})")
    print(f"           적중률  {r['acc']:.1%} (시장 {r['acc_mkt']:.1%})")
    for th in (0.0, 0.03, 0.05, 0.08, 0.12):
        st, pnl, bets = roi_from_preds(r["preds"], th)
        if st:
            print(f"  EV컷 {th:+.0%}: {st}베팅  수익 {pnl:+.2f}u  ROI {pnl/st:+.1%}  "
                  f"적중 {sum(b[3] for b in bets)}/{st}")
    print()
    _, _, bets = roi_from_preds(r["preds"], 0.05)
    print("EV>=5% 베팅 내역 (LOO 기준, 사후 아님):")
    for m, side, ev, win in bets:
        print(f"   {m:9s} {side:3s} EV{ev:+.3f}  {'적중' if win else '실패'}")
    print()
    return w


def shrink_factor():
    """
    Sample is tiny (21). Shrink model deviation toward market.
    k = n / (n + n0); n0 chosen so 21 games -> heavy shrink.
    """
    n0 = 60.0
    return N / (N + n0)


if __name__ == "__main__":
    univariate_report()
    idx, lam, name = select()
    w = final(idx, lam, name)
    k = shrink_factor()
    print("=" * 74)
    print(f"4) 표본 축소계수 (shrinkage)   n={N}, n0=60  ->  k = {k:.3f}")
    print("=" * 74)
    print("실전 적용식:")
    print(f"  logit(p) = logit(시장확률) + {k:.3f} * Σ w·x")
    print("  표본이 21경기뿐이므로 피처 신호를 1/4 수준으로만 반영 (과적합 방지).\n")
    np.save("/workspace/nhl_w.npy", np.array(w))
    with open("/workspace/nhl_model.txt", "w") as f:
        f.write(f"idx={idx}\nlam={lam}\nw={list(w)}\nk={k}\n")
    print("가중치 저장: /workspace/nhl_w.npy")
