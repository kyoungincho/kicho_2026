"""
Closing board, split along the two axes people usually conflate.

  확률 (probability) = how often this side wins       -> p_model
  가치 (value)       = how much money the bet is worth -> stake * EV

A short favorite can be 80% to win and still be worth nothing, because the
price already contains the 80%. A coin flip can be the best bet on the board.
This prints both rankings side by side so the overlap (or lack of it) is
visible, then states which list to actually bet.
"""
import numpy as np
from nhl_backtest import am_to_dec, logit, sigmoid, novig
from today_kst import (NHL, SOC, W_NHL, W_SOC, CODES, B0, B_AWAY, RAIL,
                       EV_FLOOR, EV_FLOOR_S, RAIL_S, OVR, DAY_CAP, kelly,
                       nhl_p, dog_of)

ALL = []

for mu, fav, fml, dml, feats, note in NHL:
    away, home = mu.split("@")
    fav_home = 1 if fav == home else 0
    pm, p = nhl_p(fml, dml, fav_home, feats)
    fd, dd = am_to_dec(fml), am_to_dec(dml)
    dog = dog_of(mu, fav)
    # record BOTH sides; probability and value can point at different teams
    for side, ps, dec in ((fav, p, fd), (dog, 1 - p, dd)):
        ev = ps * dec - 1
        st = kelly(ps, dec) if ev >= EV_FLOOR else 0.0
        ALL.append(dict(sport="NHL", game=mu, side=side, p=ps, dec=dec, ev=ev,
                        stake=st, val=max(0.0, st * ev * 10000), note=note,
                        floor=EV_FLOOR))

for mt, fav, fd, tg, fh, note in SOC:
    pm = min(max((1 / fd) / OVR, .02), .95)
    z = np.array([tg, 1 - tg, 1 - fh], float)
    p = float(np.clip(sigmoid(logit(pm) + z @ W_SOC),
                      max(.01, pm - RAIL_S), min(.97, pm + RAIL_S)))
    ev = p * fd - 1
    st = kelly(p, fd) if ev >= EV_FLOOR_S else 0.0
    ALL.append(dict(sport="축구", game=mt, side=fav, p=p, dec=fd, ev=ev,
                    stake=st, val=max(0.0, st * ev * 10000), note=note,
                    floor=EV_FLOOR_S))

W = 104
print("=" * W)
print("A) 확률 높은 것 — 가장 자주 이기는 쪽 (가치와 무관)")
print("=" * W)
print(f"{'순위':>3s} {'경기':10s} {'대상':5s} {'모델확률':>8s} {'배당':>5s} "
      f"{'필요확률':>8s} {'EV':>7s}  판정")
print("-" * W)
prob_rank = sorted(ALL, key=lambda r: -r["p"])[:8]
for i, r in enumerate(prob_rank, 1):
    need = 1 / r["dec"]
    verdict = "베팅 가치 있음" if r["val"] > 0 else "이기지만 돈은 안 됨"
    print(f"{i:3d} {r['game']:10s} {r['side']:5s} {r['p']*100:7.1f}% {r['dec']:5.2f} "
          f"{need*100:7.1f}% {r['ev']:+6.1%}  {verdict}")
print("-" * W)
print("'필요확률' = 배당이 요구하는 손익분기 확률. 모델확률이 이보다 높아야 돈이 된다.")

print()
print("=" * W)
print("B) 가치 있는 것 — 가격 대비 기댓값이 남는 쪽 (확률과 무관)")
print("=" * W)
val_rank = [r for r in sorted(ALL, key=lambda r: -r["val"]) if r["val"] > 0]
raw = sum(r["stake"] for r in val_rank)
sc = min(1.0, DAY_CAP / raw) if raw else 1.0
print(f"{'순위':>3s} {'경기':10s} {'대상':5s} {'모델확률':>8s} {'배당':>5s} "
      f"{'EV':>7s} {'스테이크':>8s} {'가치':>7s}")
print("-" * W)
for i, r in enumerate(val_rank, 1):
    print(f"{i:3d} {r['game']:10s} {r['side']:5s} {r['p']*100:7.1f}% {r['dec']:5.2f} "
          f"{r['ev']:+6.1%} {r['stake']*sc*100:7.2f}% {r['val']*sc:6.0f}bp")
T = sum(r["val"] for r in val_rank) * sc
print("-" * W)
print(f"총 {len(val_rank)}베팅 / 스테이크 {raw*sc*100:.1f}% / 기대 뱅크롤 +{T/100:.2f}%")

print()
print("=" * W)
print("C) 두 목록의 교집합")
print("=" * W)
pset = {(r["game"], r["side"]) for r in prob_rank}
vset = {(r["game"], r["side"]) for r in val_rank}
both = pset & vset
print(f"확률 상위 8개 중 가치까지 있는 것: {len(both)}개")
for g, s in sorted(both):
    r = next(x for x in ALL if x["game"] == g and x["side"] == s)
    print(f"  {g:10s} {s:5s} 확률 {r['p']*100:.1f}% / EV {r['ev']:+.1%}  ← 둘 다 만족")
only_p = pset - vset
print(f"\n확률만 높고 가치 없는 것: {len(only_p)}개 (사면 장기적으로 손실)")
for g, s in sorted(only_p):
    r = next(x for x in ALL if x["game"] == g and x["side"] == s)
    print(f"  {g:10s} {s:5s} 확률 {r['p']*100:5.1f}% 인데 EV {r['ev']:+6.1%} "
          f"(필요 {100/r['dec']:.1f}%)")
only_v = vset - pset
print(f"\n가치는 있는데 확률 상위권이 아닌 것: {len(only_v)}개")
for g, s in sorted(only_v, key=lambda x: -next(
        r["val"] for r in ALL if r["game"] == x[0] and r["side"] == x[1])):
    r = next(x for x in ALL if x["game"] == g and x["side"] == s)
    print(f"  {g:10s} {s:5s} 확률 {r['p']*100:5.1f}% / EV {r['ev']:+6.1%} "
          f"/ 가치 {r['val']*sc:.0f}bp")

print()
print("=" * W)
print("D) 마무리 — 어느 목록을 사야 하나")
print("=" * W)
# what happens if you bet the probability list instead of the value list
pl_ev = np.mean([r["ev"] for r in prob_rank])
vl_ev = np.mean([r["ev"] for r in val_rank])
print(f"확률 상위 8개를 1u씩 사면 평균 EV {pl_ev:+.1%}  (단위당 기대손익)")
print(f"가치 목록을 1u씩 사면   평균 EV {vl_ev:+.1%}")
print()
print("확률 목록은 '맞히는' 목록이고 가치 목록은 '버는' 목록이다. 적중률은")
print("확률 목록이 높게 나오지만 배당이 그 확률을 이미 다 먹고 있어서 평균 EV가")
print("마이너스다. 장기적으로 돈이 남는 건 가치 목록뿐이다.")
print()
print("실전 결론")
print(f"  - 돈을 넣는 대상 : 가치 목록 {len(val_rank)}개, 총 {raw*sc*100:.0f}%, 기대 +{T/100:.2f}%")
print("  - 확률 목록은 조합(축) 재료로만. 단 조합은 공제가 곱해져 EV가 더 나빠진다.")
print(f"  - 둘 다 만족하는 {len(both)}개가 오늘 보드의 유일한 '안전하면서 가치 있는' 자리다.")
