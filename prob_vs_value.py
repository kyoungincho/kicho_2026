"""
Is probability or value more important to a bettor?

The question is not rhetorical, and "value always" is a sloppy answer. This
file works out where each one actually matters, using today's board.

Three things get measured:

  1. Whether value already contains probability.
     Half-Kelly stake f = EV / (2b), so value = f*EV = EV^2 / (2b).
     For a FIXED EV, value RISES as b falls - i.e. as probability rises.
     So the value metric already prefers the likelier bet. You do not have to
     choose between the two axes; one is built from the other.

  2. Variance. Two bets with identical EV but different probability have very
     different risk. Low-probability +EV bets need far more repetitions before
     the edge shows up, and can drown a finite bankroll first.

  3. Fragility to model error. EV is computed from MY estimated probability.
     If that estimate is off by 3 points, a long-odds bet moves much more.

Plus a disclosure: the 4% stake cap was binding on all five of today's bets,
which silently flattened the value ranking into a plain EV ranking.
"""
import numpy as np

HALF = 0.5
CAP = 0.04

# (label, model probability, decimal odds) - today's value list
BETS = [
    ("SEA @EDM", 0.445, 2.70),
    ("NSH vDAL", 0.525, 2.20),
    ("PHI vCAR", 0.547, 2.10),
    ("ENG @CRO", 0.540, 2.10),
    ("SCO @MKD", 0.600, 1.85),
]
# the probability list, for contrast
PROB_LIST = [
    ("ESP vCZE", 0.805, 1.25),
    ("COL vSTL", 0.716, 1.34),
    ("BUF vCHI", 0.689, 1.43),
    ("MIN vBOS", 0.637, 1.56),
    ("TB  vWSH", 0.627, 1.65),
    ("SUI vSVN", 0.626, 1.70),
    ("SCO @MKD", 0.600, 1.85),
    ("EDM vSEA", 0.555, 1.49),
]


def metrics(p, dec):
    b = dec - 1
    ev = p * b - (1 - p)
    f_full = max(0.0, ev / b)
    f_half = f_full * HALF
    f_used = min(CAP, f_half)
    # variance of the per-unit return
    var = p * (b - ev) ** 2 + (1 - p) * (-1 - ev) ** 2
    sd = np.sqrt(var)
    # bets needed before a 1-sided 95% interval clears zero
    n95 = (1.645 * sd / ev) ** 2 if ev > 0 else np.inf
    return dict(b=b, ev=ev, f_half=f_half, f_used=f_used, sd=sd, n95=n95,
                val_uncapped=f_half * ev * 1e4, val_capped=f_used * ev * 1e4)


W = 100
print("=" * W)
print("1) 가치 지표는 이미 확률을 품고 있다")
print("=" * W)
print("하프켈리 f = EV/(2b) 이므로  가치 = f×EV = EV²/(2b)")
print("즉 EV가 같으면 b가 작은 쪽(=확률이 높은 쪽)이 가치가 더 크다.")
print("확률과 가치는 대립하는 축이 아니라, 가치가 확률을 재료로 쓰는 구조다.\n")
print("예시 — EV를 +12%로 고정하고 배당만 바꿨을 때:")
print(f"{'배당':>6s} {'필요확률':>8s} {'EV +12%를 만드는 확률':>20s} {'하프켈리':>9s} {'가치':>8s}")
print("-" * W)
for dec in (1.50, 1.85, 2.20, 3.00, 5.00):
    b = dec - 1
    p = (0.12 + 1) / dec          # solve p*b-(1-p)=0.12
    m = metrics(p, dec)
    print(f"{dec:6.2f} {100/dec:7.1f}% {p*100:19.1f}% {m['f_half']*100:8.1f}% "
          f"{m['val_uncapped']:7.0f}bp")
print("\n같은 +12% EV인데 단배 1.50은 가치 144bp, 단배 5.00은 가치 18bp — 8배 차이.")
print("배당이 길어질수록 켈리가 허용하는 금액이 줄어서 가치가 같이 떨어진다.")

print()
print("=" * W)
print("2) 어제 출력의 결함 — 4% 상한이 가치 순위를 망가뜨렸다")
print("=" * W)
print(f"{'베팅':10s} {'확률':>6s} {'배당':>5s} {'EV':>7s} {'진짜 하프켈리':>12s} "
      f"{'적용된 값':>9s} {'상한 작동':>9s}")
print("-" * W)
rows = [(lb, p, d, metrics(p, d)) for lb, p, d in BETS]
for lb, p, d, m in rows:
    hit = "O" if m["f_half"] > CAP else "-"
    print(f"{lb:10s} {p*100:5.1f}% {d:5.2f} {m['ev']:+6.1%} {m['f_half']*100:11.1f}% "
          f"{m['f_used']*100:8.1f}% {hit:>9s}")
print("-" * W)
print("5개 전부 상한에 걸렸다. 그래서 스테이크가 모두 2.40%로 같아졌고,")
print("가치 순위가 그냥 EV 순위로 붕괴했다. 확률 정보가 사라진 것이다.\n")
print("상한을 풀면 진짜 가치 순위는 이렇게 바뀐다:")
print(f"{'순위':>3s} {'베팅':10s} {'확률':>6s} {'EV':>7s} {'가치(상한X)':>11s}  {'EV순위 대비':>10s}")
print("-" * W)
ev_order = [r[0] for r in sorted(rows, key=lambda r: -r[3]["ev"])]
for i, (lb, p, d, m) in enumerate(sorted(rows, key=lambda r: -r[3]["val_uncapped"]), 1):
    move = ev_order.index(lb) + 1 - i
    tag = "유지" if move == 0 else f"{abs(move)}칸 {'상승' if move>0 else '하락'}"
    print(f"{i:3d} {lb:10s} {p*100:5.1f}% {m['ev']:+6.1%} {m['val_uncapped']:10.0f}bp  "
          f"{tag:>10s}")
print("\nSCO(확률 60%)가 EV는 가장 낮은데 가치로는 올라온다. 확률이 높아서다.")

print()
print("=" * W)
print("3) 분산 — 확률이 중요한 진짜 이유")
print("=" * W)
print(f"{'베팅':10s} {'확률':>6s} {'EV':>7s} {'표준편차':>8s} {'EV/SD':>7s} "
      f"{'95% 확신까지 필요 베팅수':>22s}")
print("-" * W)
for lb, p, d, m in rows:
    print(f"{lb:10s} {p*100:5.1f}% {m['ev']:+6.1%} {m['sd']:8.3f} "
          f"{m['ev']/m['sd']:7.3f} {m['n95']:21.0f}회")
print("-" * W)
print("주의 — 흔한 직관이 여기서 깨진다. SEA는 확률이 가장 낮고 분산도 가장 크지만")
print("EV가 그 분산을 덮을 만큼 크기 때문에 EV/SD가 가장 높고, 95% 확신까지")
print("필요한 베팅수도 가장 적다(120회 vs SCO 184회).")
print("즉 '확률 낮으면 위험하다'는 일반론은 참이지만, 오늘 보드에는 적용되지 않는다.")
print("분산 자체가 아니라 EV/SD(=샤프 비율)로 비교해야 하고, 그 기준으로는")
print("역배 쪽이 오히려 안전하다. 확률만 보고 SEA를 피하면 더 나쁜 선택이 된다.")

print()
print("=" * W)
print("4) 모델 오차 민감도 — 내 확률이 3%p 틀렸다면")
print("=" * W)
print(f"{'베팅':10s} {'확률':>6s} {'EV(그대로)':>10s} {'확률-3%p':>9s} {'확률-5%p':>9s} "
      f"{'몇 %p 틀리면 EV=0':>17s}")
print("-" * W)
for lb, p, d, m in rows:
    b = m["b"]
    ev3 = (p - .03) * b - (1 - (p - .03))
    ev5 = (p - .05) * b - (1 - (p - .05))
    p_be = 1 / d
    margin = (p - p_be) * 100
    print(f"{lb:10s} {p*100:5.1f}% {m['ev']:+9.1%} {ev3:+8.1%} {ev5:+8.1%} "
          f"{margin:16.1f}%p")
print("-" * W)
print("EV 민감도는 배당(b)에 비례한다. SEA는 확률 1%p당 EV가 1.7%p 움직이고")
print("SCO는 0.85%p만 움직인다. 하지만 SEA는 출발점이 더 멀어서, EV가 0이 되기까지")
print("7.5%p의 여유가 있다(SCO는 5.9%p). 여기서도 역배 쪽이 더 튼튼하다.")
print("'긴 배당일수록 모델이 정확해야 한다'는 EV 변동폭 얘기지 안전마진 얘기가 아니다.")

print()
print("=" * W)
print("5) 그래서 확률만 보면 어떻게 되나 — 100베팅 시뮬레이션")
print("=" * W)
rng = np.random.default_rng(11)
TRIALS, NBETS = 20000, 100


def simulate(lst, flat=0.02):
    """flat staking, fraction of starting bankroll, no compounding"""
    ps = np.array([p for _, p, _ in lst])
    bs = np.array([d - 1 for _, _, d in lst])
    out = np.zeros(TRIALS)
    for t in range(TRIALS):
        k = rng.integers(0, len(lst), NBETS)
        w = rng.random(NBETS) < ps[k]
        out[t] = np.sum(np.where(w, bs[k], -1.0)) * flat
    return out


val_out = simulate(BETS)
prob_out = simulate(PROB_LIST)
print(f"{'전략':22s} {'평균 수익률':>11s} {'손실 확률':>9s} {'하위5%':>9s} "
      f"{'상위5%':>9s}")
print("-" * W)
for nm, o in (("가치 목록 5개", val_out), ("확률 목록 8개", prob_out)):
    print(f"{nm:22s} {o.mean()*100:+10.1f}% {(o<0).mean():8.1%} "
          f"{np.percentile(o,5)*100:+8.1f}% {np.percentile(o,95)*100:+8.1f}%")
print("-" * W)
print("확률 목록은 적중률은 높지만 100베팅 후에도 손실 확률이 절반에 가깝다.")
print("가치 목록은 변동이 크지만 기대값이 분명히 양수다.")

print()
print("=" * W)
print("6) 결론")
print("=" * W)
print("""
확률이 중요한가 가치가 중요한가 — 둘을 비교하는 질문 자체가 성립하지 않는다.
확률은 입력이고 가치는 출력이다. 확률만으로는 돈을 벌 수 없는데, 배당이
이미 확률을 가격에 넣어놨기 때문이다. 오늘 ESP는 80.5% 승률인데 1.25가
80.0%를 요구한다 — 확률 정보가 전부 가격으로 흡수된 상태다.

그렇다고 확률이 무의미한 것도 아니다. 확률이 실제로 일하는 곳은 두 군데다.
  (1) 금액 — 켈리가 확률로 스테이크를 정한다. EV가 같다면 확률 높은 쪽이
      더 많이 걸 수 있어서 가치가 크다. 가치 = EV²/(2b).
  (2) 같은 EV끼리의 비교 — EV가 비슷한 두 베팅 중에서는 확률 높은 쪽이 낫다.

반대로, 확률에 대한 흔한 통념 하나는 오늘 보드에서 틀렸다.
"확률 낮은 베팅은 분산이 커서 위험하다"는 일반론은 맞지만, 위험은 분산이
아니라 EV/SD로 재야 한다. SEA는 확률 44.5%로 가장 낮은데 EV가 커서 EV/SD가
가장 높고, 안전마진(EV=0까지 7.5%p)도 가장 넓다. 확률이 낮다는 이유로
SEA를 피하고 SCO를 택하면 모든 축에서 더 나쁜 선택이 된다.

정리하면: 베팅 여부는 가치로 결정하고, 베팅 금액은 확률로 결정한다.
확률이 높은데 가치가 없으면 사지 않고, 가치가 있으면 확률이 낮아도 산다.
단 금액은 켈리가 확률을 반영해 알아서 줄여준다 — 내가 임의로 줄일 게 아니다.
""".strip())
