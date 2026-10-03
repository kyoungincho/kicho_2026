"""
STL @ COL deep dive (KST 10/4).

The model passes on this game, and the reason is unusually clean: the two
forces in it point in opposite directions and cancel almost exactly.

  - the global favorite tax says fade COL      (-0.103 logit)
  - STL on the second of a back-to-back says back COL (+0.096 logit)

Net shift is -0.007. So the model lands essentially on top of the market, and
neither side clears its EV floor.

Beyond confirming the pass, this reverse-engineers what WOULD be needed to make
either side bettable, which is the useful part: at -298 the answer is "much
more than this game has".
"""
import numpy as np
from nhl_backtest import GAMES, am_to_dec, logit, sigmoid, novig

W_NHL = np.load("/workspace/W_v4.npy")
CODES = ["fs", "ds", "fg", "dg", "fb", "db", "rl", "up"]
WM = dict(zip(CODES, W_NHL))
B0, B_AWAY, RAIL = -0.1031, -0.050, 0.09
EV_FLOOR, DP_FLOOR = 0.04, 0.04

FAV_ML, DOG_ML = -298, +240
fav_dec, dog_dec = am_to_dec(FAV_ML), am_to_dec(DOG_ML)
pm = novig(FAV_ML, DOG_ML)
lm = logit(pm)

W = 92
print("=" * W)
print("STL @ COL  —  COL -298 / STL +240  (콜로라도 홈)")
print("=" * W)
print(f"COL 단배 {fav_dec:.3f} (암시 {1/fav_dec:.1%})  /  "
      f"STL 단배 {dog_dec:.2f} (암시 {1/dog_dec:.1%})")
print(f"공제 {1/fav_dec + 1/dog_dec - 1:.1%}  →  무공제 COL {pm:.1%} / STL {1-pm:.1%}")
print(f"손익분기: COL은 {1/fav_dec:.1%} 넘겨야, STL은 {1/dog_dec:.1%} 넘겨야 돈이 된다.")

print()
print("=" * W)
print("1) 모델 계산 — 두 힘이 정확히 상쇄된다")
print("=" * W)
steps = [
    ("시장 무공제", 0.0, "출발점"),
    ("전역항: 페이브세", B0, "정규리그 페이브는 암시보다 2.5%p 덜 이긴다"),
    ("STL B2B 뒷경기 (db)", WM["db"], "원정 백투백 = 역배에 불리 = COL에 유리"),
]
cur = lm
print(f"{'단계':24s} {'logit 변화':>10s} {'COL 확률':>9s}  설명")
print("-" * W)
for nm, d, why in steps:
    cur += d
    print(f"{nm:24s} {d:+10.3f} {sigmoid(cur)*100:8.1f}%  {why}")
p_model = float(np.clip(sigmoid(cur), pm - RAIL, pm + RAIL))
print("-" * W)
print(f"순 변화 {cur - lm:+.3f} logit  →  모델 COL {p_model:.1%} (시장 {pm:.1%}, "
      f"Δp {(p_model-pm)*100:+.1f}%p)")
print("\n페이브세 -0.103과 B2B +0.096이 거의 같은 크기로 맞서서 서로를 지운다.")
print("모델이 시장에 '할 말이 없다'는 상태이고, 이게 PASS의 이유다.")

print()
print("=" * W)
print("2) 양쪽 EV — 둘 다 산다")
print("=" * W)
ev_f = p_model * fav_dec - 1
ev_d = (1 - p_model) * dog_dec - 1
print(f"{'측':6s} {'배당':>6s} {'모델확률':>8s} {'필요확률':>8s} {'EV':>8s} {'Δp':>7s}  판정")
print("-" * W)
for nm, p, dec in (("COL", p_model, fav_dec), ("STL", 1 - p_model, dog_dec)):
    ev = p * dec - 1
    dp = abs(p_model - pm) * 100
    print(f"{nm:6s} {dec:6.2f} {p*100:7.1f}% {100/dec:7.1f}% {ev:+7.1%} {dp:6.1f}%p  "
          f"{'베팅' if ev >= EV_FLOOR and dp >= DP_FLOOR*100 else 'PASS'}")
print("-" * W)
print(f"COL은 {1/fav_dec:.1%}가 필요한데 모델은 {p_model:.1%} → {ev_f:+.1%}")
print(f"STL은 {1/dog_dec:.1%}가 필요한데 모델은 {1-p_model:.1%} → {ev_d:+.1%}")
print(f"Δp도 {abs(p_model-pm)*100:.1f}%p로 하한 4%p에 한참 못 미친다. "
      f"이중 하한 양쪽 모두 불통과.")

print()
print("=" * W)
print("3) 역산 — 뭐가 있어야 베팅이 되나")
print("=" * W)


def need_shift(target_p):
    return logit(target_p) - lm


col_need_p = (1 + EV_FLOOR) / fav_dec
stl_need_p = (1 + EV_FLOOR) / dog_dec
col_need_dp = pm + DP_FLOOR
stl_need_colp = pm - DP_FLOOR

col_target = max(col_need_p, col_need_dp)
stl_target_col = min(1 - stl_need_p, stl_need_colp)

print(f"COL을 사려면 : 모델 COL >= {col_target:.1%}  "
      f"(EV컷 {col_need_p:.1%} / Δp컷 {col_need_dp:.1%} 중 큰 쪽)")
print(f"              필요 logit 이동 {need_shift(col_target):+.3f}, "
      f"전역항 제외 피처가 {need_shift(col_target)-B0:+.3f} 를 만들어야 함")
print(f"STL을 사려면 : 모델 COL <= {stl_target_col:.1%}")
print(f"              필요 logit 이동 {need_shift(stl_target_col):+.3f}, "
      f"피처가 {need_shift(stl_target_col)-B0:+.3f} 를 만들어야 함")
print(f"레일 한계    : COL {pm-RAIL:.1%} ~ {pm+RAIL:.1%} (±9%p)")

print()
print("실제 피처 조합별로 계산:")
SCEN = [
    ("현재 (STL B2B만)", dict(db=1)),
    ("+ STL 백업골리 확정", dict(db=1, dg=1)),
    ("+ STL 주력 1명 결장", dict(db=1, dg=1, ds=1)),
    ("+ STL 주력 2명 결장", dict(db=1, dg=1, ds=2)),
    ("반대: COL 주력 1명 OUT (B2B 유지)", dict(db=1, fs=1)),
    ("반대: COL 주력1+골리 (B2B 유지)", dict(db=1, fs=1, fg=1)),
    ("반대: COL 주력1+골리, STL B2B 아님", dict(fs=1, fg=1)),
]
print("엣지 1/2 축소(채택 전략)까지 적용한 값으로 판정한다.")
print(f"{'시나리오':34s} {'모델COL':>8s} {'축소COL':>8s} {'COL EV':>8s} "
      f"{'STL EV':>8s}  판정")
print("-" * W)
for nm, fe in SCEN:
    z = lm + B0 + sum(WM[c] * v for c, v in fe.items())
    p = float(np.clip(sigmoid(z), pm - RAIL, pm + RAIL))
    psh = pm + 0.5 * (p - pm)                 # adopted: believe half the edge
    e_f, e_d = psh * fav_dec - 1, (1 - psh) * dog_dec - 1
    dp = abs(psh - pm)
    buy = "PASS"
    if e_f >= EV_FLOOR and dp >= DP_FLOOR:
        buy = "COL 베팅"
    elif e_d >= EV_FLOOR and dp >= DP_FLOOR:
        buy = "STL 베팅"
    print(f"{nm:34s} {p*100:7.1f}% {psh*100:7.1f}% {e_f:+7.1%} {e_d:+7.1%}  {buy}")
print("-" * W)
print("엣지를 절반만 믿으면 Δp 하한 4%p가 사실상 '모델 Δp 8%p 이상'을 요구한다.")
print("레일이 ±9%p이므로, 베팅 가능 구간은 레일 거의 끝까지 간 경우뿐이다.")
print("결론 두 개:")
print(" (1) COL 쪽은 어떤 조합으로도 못 산다. STL이 B2B + 백업골리 + 주력 2명")
print("     결장까지 전부 쌓여도 축소 EV가 +0.6%다. -298이 그만큼 짧다.")
print("     '콜로라도가 이긴다'는 예상과 'COL -298을 산다'는 완전히 다른 얘기다.")
print(" (2) STL 쪽은 COL에 구멍이 생기면 열린다. 단 STL이 B2B라서 그 효과가")
print("     상쇄되어, COL 주력+골리 둘 다 빠지고도 축소 EV +7.8%로 컷에 못 미친다.")
print("     이 경기에서 뜬 유일한 피처가 역배에 불리하게 작용하는 구조다.")

print()
print("=" * W)
print("4) 백테스트 비교군 — 초크 -170 이하 7경기")
print("=" * W)
print(f"{'경기':10s} {'페이브ML':>8s} {'페이브 구멍':>11s} {'역배 구멍':>10s} {'결과':>6s}")
print("-" * W)
heavy = [g for g in GAMES if g[3] <= -170]
for g in heavy:
    fh = g[6] + g[8]          # fs + fg
    dh = g[7] + g[9] + g[11]  # ds + dg + db
    print(f"{g[1]:10s} {g[3]:8d} {fh:11.0f} {dh:10.0f} "
          f"{'페이브승' if g[5] else '역배승':>6s}")
print("-" * W)
hw = sum(g[5] for g in heavy)
print(f"전체 {hw}/{len(heavy)} = {hw/len(heavy):.1%}  (평균 암시 "
      f"{np.mean([novig(g[3],g[4]) for g in heavy]):.1%})")
sub_fh = [g for g in heavy if (g[6] + g[8]) > 0]
sub_dh = [g for g in heavy if (g[6] + g[8]) == 0 and (g[7] + g[9] + g[11]) > 0]
print(f"  페이브에 구멍 있음 : {sum(g[5] for g in sub_fh)}/{len(sub_fh)}")
print(f"  역배에만 구멍 있음 : {sum(g[5] for g in sub_dh)}/{len(sub_dh)}")
print("\n오늘 COL-STL은 '역배에만 구멍' 쪽(역배 B2B)이고, 그 표본에서는")
print("페이브가 잘 이겼다. 그런데 -298은 표본 최고가(-270)보다도 비싸서,")
print("'이길 것 같다'와 '가격이 맞다'가 분리된다. 백테스트가 가격을 정당화하지 않는다.")

print()
print("=" * W)
print("5) 퍽라인은?")
print("=" * W)
print("COL -1.5 / STL +1.5 는 이번 최적화에 포함되지 않았다.")
print("퍽라인 백테스트를 하지 않았으므로 검증된 가중치가 없고, EV를 계산할 근거도 없다.")
print("ML 모델을 퍽라인에 전용하는 것도 안 된다 — 승패 확률과 2점차 커버 확률은")
print("상관은 높지만 같은 분포가 아니고, 특히 빈네트 상황에서 크게 갈라진다.")
print("→ 퍽라인도 PASS. 숫자 없이 추천하면 지금까지 세운 규칙을 내가 어기는 것이다.")

print()
print("=" * W)
print("6) 최종")
print("=" * W)
print(f"""
판정: PASS (양방향 모두)

  COL -298 : 모델 {p_model:.1%} vs 필요 {1/fav_dec:.1%}  →  EV {ev_f:+.1%}
  STL +240 : 모델 {1-p_model:.1%} vs 필요 {1/dog_dec:.1%}  →  EV {ev_d:+.1%}
  Δp {abs(p_model-pm)*100:.1f}%p (하한 4%p 미달)

이 경기는 '모르는 경기'가 아니라 '시장이 이미 맞게 매긴 경기'다.
페이브세(-0.103)와 STL 백투백(+0.096)이 서로를 지워서 모델이 시장과 같은 자리에
선다. 그러면 남는 건 공제뿐이고, 공제를 내고 사는 베팅은 손실이다.

이 경기는 '모르는 경기'가 아니라 '시장이 이미 맞게 매긴 경기'다. 그러면 남는 건
공제뿐이고, 공제를 내고 사는 베팅은 손실이다.

경기 전 판정이 바뀌는 조건 (엣지 1/2 축소 기준):
  STL을 사게 되는 경우 - COL 주력 결장 + 골리 다운그레이드가 둘 다 확정되고,
    거기에 STL의 백투백 불리가 상쇄되지 않을 정도여야 한다. 상당히 빡빡하다.
  COL을 사게 되는 경우 - 없다. 가격이 이미 모든 호재를 먹었다.

즉 이 경기는 '발표를 기다릴 가치가 있는 경기'도 아니다. 다른 경기를 보는 게 맞다.
""".strip())
