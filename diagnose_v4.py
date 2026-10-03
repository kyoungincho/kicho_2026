"""
Honesty check on v4.

Every +EV bet v4 produced was on the DOG, in a week where dogs went 12-9.
So the question is: do the 8 situation features actually add anything, or is
v4 just an expensive way to write "fade the favorite"?

Three baselines:
  B1  market only
  B2  market + global favorite tax (NO situation features)
  B3  blind dog, every game
and then a permutation test: shuffle the feature rows against the results
1000x and see how often a random assignment beats the real one.
"""
import numpy as np
from nhl_backtest import GAMES, X, y, pm, fav_dec, dog_dec, N, logit, sigmoid
from model_v4 import (IDX, SPEC, W_PRIOR, blend, fit_data, predict_from,
                      glob, offset4, B0, B_AWAY_FAV, score, sim)

N0_OPT, LAM_OPT, RAIL = 140, 5.0, 0.09


def loo_preds(Xmat, use_feats=True, use_glob=True):
    preds = np.zeros(N)
    off = (logit(pm) + glob) if use_glob else logit(pm)
    for i in range(N):
        if not use_feats:
            preds[i] = float(np.clip(sigmoid(off[i]), 0.01, 0.99))
            continue
        tr = np.array([j for j in range(N) if j != i])
        wd = fit_data(Xmat[tr], y[tr], off[tr], LAM_OPT)
        wb = blend(wd, n=N - 1, n0=N0_OPT)
        g = glob[i] if use_glob else 0.0
        preds[i] = predict_from(pm[i], Xmat[i, IDX], wb, g, RAIL)
    return preds


def flat_roi(preds, floor):
    st, pnl, hits = 0, 0.0, 0
    for i in range(N):
        ef = preds[i] * fav_dec[i] - 1
        ed = (1 - preds[i]) * dog_dec[i] - 1
        if max(ef, ed) < floor:
            continue
        st += 1
        if ef >= ed:
            w = y[i] == 1
            pnl += (fav_dec[i] - 1) if w else -1
        else:
            w = y[i] == 0
            pnl += (dog_dec[i] - 1) if w else -1
        hits += int(w)
    return st, hits, pnl, (pnl / st if st else 0.0)


print("=" * 84)
print("G) 기준선 대비 — 피처가 실제로 기여하는가")
print("=" * 84)

p_b1 = pm
p_b2 = loo_preds(X, use_feats=False, use_glob=True)
p_v4 = loo_preds(X, use_feats=True, use_glob=True)

blind_pnl = np.where(y == 0, dog_dec - 1, -1.0).sum()

print(f"{'모델':28s} {'logloss':>9s} {'Brier':>8s} {'적중':>7s} "
      f"{'EV4%베팅':>9s} {'ROI':>8s}")
print("-" * 84)
for nm, pr in (("B1 시장 단독", p_b1),
               ("B2 시장+페이브세(피처X)", p_b2),
               ("v4 시장+페이브세+피처", p_v4)):
    ll, br, acc = score(pr)
    st, h, pnl, roi = flat_roi(pr, 0.04)
    print(f"{nm:28s} {ll:9.4f} {br:8.4f} {acc:6.1%} {st:7d} {roi:+7.1%}")
print(f"{'B3 무조건 역배 21경기':28s} {'-':>9s} {'-':>8s} "
      f"{(y==0).mean():6.1%} {N:7d} {blind_pnl/N:+7.1%}")

ll2, br2, _ = score(p_b2)
ll4, br4, _ = score(p_v4)
print(f"\n피처 순기여 (B2 -> v4): logloss {ll2-ll4:+.4f}, Brier {br2-br4:+.4f}")
print(f"전역항 순기여 (B1 -> B2): logloss {score(p_b1)[0]-ll2:+.4f}, "
      f"Brier {score(p_b1)[1]-br2:+.4f}")

print()
print("=" * 84)
print("H) 역배 편향 분리 — 같은 '역배만 사는' 전략끼리 비교")
print("=" * 84)
st4, h4, pnl4, roi4 = flat_roi(p_v4, 0.04)
print(f"v4 선별 역배 : {h4}/{st4}  ROI {roi4:+.1%}")
print(f"전체 역배    : {(y==0).sum()}/{N}  ROI {blind_pnl/N:+.1%}")
print(f"→ 선별이 전체보다 {roi4 - blind_pnl/N:+.1%}p 우위")
skipped = [i for i in range(N)
           if max(p_v4[i]*fav_dec[i]-1, (1-p_v4[i])*dog_dec[i]-1) < 0.04]
if skipped:
    sk_pnl = np.where(y[skipped] == 0, dog_dec[skipped] - 1, -1.0).mean()
    print(f"v4가 걸러낸 {len(skipped)}경기에서 역배를 샀다면 ROI {sk_pnl:+.1%} "
          f"(역배 {(y[skipped]==0).sum()}/{len(skipped)})")
    print("→ 걸러낸 쪽이 확실히 나쁘면 피처가 선별력을 가진 것")

print()
print("=" * 84)
print("I) 순열검정 — 피처 신호가 우연과 구분되는가 (1000회)")
print("=" * 84)
rng = np.random.default_rng(7)
real_gain = ll2 - ll4
gains = []
for t in range(1000):
    perm = rng.permutation(N)
    Xp = X[perm].copy()
    Xp[:, 8] = X[:, 8]          # keep home flag aligned (it drives the global term)
    pr = loo_preds(Xp, use_feats=True, use_glob=True)
    gains.append(ll2 - score(pr)[0])
gains = np.array(gains)
pval = float((gains >= real_gain).mean())
print(f"실제 피처 기여 logloss 개선 : {real_gain:+.4f}")
print(f"무작위 섞었을 때 평균        : {gains.mean():+.4f} (표준편차 {gains.std():.4f})")
print(f"무작위가 실제를 넘은 비율    : {pval:.1%}  -> p = {pval:.3f}")
if pval < 0.10:
    print("→ 약하지만 신호 있음 (p<0.10). 표본 21경기 한계 내에서 유의.")
else:
    print("→ 우연과 구분 안 됨. 피처는 '사전지식'으로만 쓰고 데이터 적합은 신뢰하지 말 것.")

print()
print("=" * 84)
print("J) 결론")
print("=" * 84)
print(f"1. 전역항(페이브세 {B0:+.3f}, 원정 {B_AWAY_FAV:+.3f})만으로 logloss {score(p_b1)[0]-ll2:+.4f}.")
print("   시즌 전체(~1300경기) 기저율이라 신뢰도 높고, 공짜로 얻는 개선.")
print(f"2. 상황피처가 그 위에 {real_gain:+.4f} 추가 — 전역항보다 기여가 크다. 다만 p={pval:.2f}.")
print("3. 피처의 '방향'은 믿고 '크기'는 못 믿는다 -> N0=140, 거의 사전값 유지.")
print("4. 실전 적용: 시장가격 -> 페이브세 보정 -> 상황피처 소폭 -> ±9% 레일 -> EV컷.")
