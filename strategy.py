"""
Betting strategy, derived rather than asserted.

The governing fact is that the edge is NOT established. The permutation test on
the situational features came back p = 0.080 on 21 games. That is "probably
something", not "proven". Any staking plan that assumes the model is right is
the wrong plan.

So every staking scheme is simulated under four worlds:

  A  edge as modeled      true p = model p
  B  edge half as big     true p = market + 0.5*(model - market)
  C  no edge              true p = market no-vig p   (you are just paying vig)
  D  edge is backwards    true p = market - (model - market)

A plan is only acceptable if it grows well in A/B and SURVIVES C/D, because
C and D are live possibilities at p = 0.080.

Finally a sequential monitor: after how many losing bets should the model be
switched off?
"""
import numpy as np

# today's board: (label, market no-vig p, model p, decimal)
BOARD = [
    ("SEA", 0.355, 0.445, 2.70),
    ("NSH", 0.437, 0.525, 2.20),
    ("PHI", 0.457, 0.547, 2.10),
    ("ENG", 0.449, 0.540, 2.10),
    ("SCO", 0.510, 0.600, 1.85),
]

W = 100
print("=" * W)
print("0) 전제 — 엣지는 '증명'되지 않았다")
print("=" * W)
print("순열검정 p=0.080 (21경기). 방향은 맞을 가능성이 높지만 확정이 아니다.")
print("그리고 오늘 5베팅 전부 ±9% 레일에 걸려 있다 = 모델이 더 가고 싶어했다는 뜻이고,")
print("레일이 없으면 과신이 그대로 금액으로 나갔을 상황이다.")
print()
print(f"{'베팅':6s} {'시장%':>7s} {'모델%':>7s} {'편차':>6s} {'레일':>6s}")
print("-" * W)
for lb, pm, pmo, d in BOARD:
    print(f"{lb:6s} {pm*100:6.1f}% {pmo*100:6.1f}% {(pmo-pm)*100:+5.1f}p "
          f"{'한계' if abs(pmo-pm) >= 0.088 else '여유':>6s}")

# ---------------------------------------------------------------- worlds
def world_p(pm, pmo, mode):
    if mode == "A":
        return pmo
    if mode == "B":
        return pm + 0.5 * (pmo - pm)
    if mode == "C":
        return pm
    return pm - (pmo - pm)


# ---------------------------------------------------------------- staking
def stake_fn(name):
    """Stake as a fraction of bankroll.

    The '엣지축소' schemes do not shrink the stake after the fact; they shrink
    the believed edge first and then size Kelly honestly on that. That is the
    standard way to bet when the model itself is uncertain, and it behaves very
    differently from just trimming the stake.
    """
    def f(i, dec):
        b = dec - 1
        pmo, pm = pmo_a[i], pm_a[i]

        def kel(p):
            return max(0.0, (p * b - (1 - p)) / b)

        if name == "플랫 1%":
            return 0.01
        if name == "플랫 2%":
            return 0.02
        if name == "1/4켈리":
            return min(0.04, kel(pmo) * 0.25)
        if name == "1/2켈리 상한4%":
            return min(0.04, kel(pmo) * 0.5)
        if name == "1/2켈리 상한8%":
            return min(0.08, kel(pmo) * 0.5)
        if name == "풀켈리":
            return min(0.25, kel(pmo))
        if name == "엣지1/2+하프켈리":
            return min(0.04, kel(pm + 0.5 * (pmo - pm)) * 0.5)
        if name == "엣지1/2+쿼터켈리":
            return min(0.04, kel(pm + 0.5 * (pmo - pm)) * 0.25)
        if name == "엣지1/3+하프켈리":
            return min(0.04, kel(pm + (pmo - pm) / 3) * 0.5)
        raise ValueError(name)
    return f


SCHEMES = ["플랫 1%", "플랫 2%", "1/4켈리", "1/2켈리 상한4%", "1/2켈리 상한8%",
           "풀켈리", "엣지1/2+하프켈리", "엣지1/2+쿼터켈리", "엣지1/3+하프켈리"]
NBETS, TRIALS = 150, 20000
rng = np.random.default_rng(3)

pm_a = np.array([b[1] for b in BOARD])
pmo_a = np.array([b[2] for b in BOARD])
dec_a = np.array([b[3] for b in BOARD])


def simulate(scheme, mode):
    f = stake_fn(scheme)
    stakes = np.array([f(i, dec_a[i]) for i in range(len(BOARD))])
    tp = np.array([world_p(pm_a[i], pmo_a[i], mode) for i in range(len(BOARD))])
    b_a = dec_a - 1
    finals = np.zeros(TRIALS)
    maxdd = np.zeros(TRIALS)
    for t in range(TRIALS):
        k = rng.integers(0, len(BOARD), NBETS)
        wins = rng.random(NBETS) < tp[k]
        mult = 1.0 + stakes[k] * np.where(wins, b_a[k], -1.0)
        path = np.cumprod(mult)
        finals[t] = path[-1]
        peak = np.maximum.accumulate(np.concatenate([[1.0], path]))
        maxdd[t] = np.min(np.concatenate([[1.0], path]) / peak)
    return finals, maxdd


print()
print("=" * W)
print(f"1) 사이징별 결과 — {NBETS}베팅 복리, 4개 세계에서 각 {TRIALS:,}회 시뮬")
print("=" * W)
store = {}
for mode, title in (("A", "A 엣지 그대로"), ("B", "B 엣지 절반"),
                    ("C", "C 엣지 없음"), ("D", "D 엣지 반대")):
    print(f"\n[{title}]")
    print(f"{'사이징':16s} {'중위 뱅크롤':>11s} {'평균':>8s} {'수익확률':>8s} "
          f"{'반토막 확률':>11s} {'중위 최대낙폭':>13s}")
    print("-" * W)
    for s in SCHEMES:
        fin, dd = simulate(s, mode)
        store[(s, mode)] = (fin, dd)
        print(f"{s:16s} {np.median(fin):10.2f}x {fin.mean():7.2f}x "
              f"{(fin>1).mean():7.1%} {(fin<0.5).mean():10.1%} "
              f"{(1-np.median(dd))*100:12.1f}%")

print()
print("=" * W)
print("2) 판정 — 세계별 확률을 부여한 베이즈 가중")
print("=" * W)
print("4개 세계를 동등 취급하면 지나치게 비관적이다. 순열검정 p=0.080은")
print("'엣지 없음'이 8% 수준으로 가능하다는 뜻이고, '엣지 반대'는 그보다 드물다.")
print("기준은 기대 로그성장(=켈리가 최대화하는 양) 최대화, 단 반토막 확률 제약.\n")
PRIORS = {"A": 0.25, "B": 0.40, "C": 0.28, "D": 0.07}
print("세계 확률: " + " / ".join(f"{k} {v:.0%}" for k, v in PRIORS.items()))
print()
print(f"{'사이징':16s} {'기대 로그성장':>12s} {'가중 중위':>9s} {'가중 반토막':>11s} "
      f"{'가중 수익확률':>12s}  판정")
print("-" * W)
verdict = []
for s in SCHEMES:
    elog = sum(PRIORS[m] * np.mean(np.log(np.maximum(store[(s, m)][0], 1e-6)))
               for m in PRIORS)
    med = sum(PRIORS[m] * np.median(store[(s, m)][0]) for m in PRIORS)
    ruin = sum(PRIORS[m] * (store[(s, m)][0] < 0.5).mean() for m in PRIORS)
    pwin = sum(PRIORS[m] * (store[(s, m)][0] > 1).mean() for m in PRIORS)
    ok = ruin < 0.10
    verdict.append((s, elog, med, ruin, pwin, ok))
    print(f"{s:16s} {elog:+11.4f} {med:8.2f}x {ruin:10.1%} {pwin:11.1%}  "
          f"{'적격' if ok else '탈락(파산위험)'}")
print("-" * W)
elig = [v for v in verdict if v[5]]
PICK = max(elig, key=lambda v: v[1]) if elig else max(verdict, key=lambda v: v[1])
print(f"채택: {PICK[0]}  (기대 로그성장 {PICK[1]:+.4f}, 반토막 {PICK[3]:.1%})")

print()
print("세계 확률을 바꿔도 결론이 유지되는지 (민감도):")
for nm, pr in (("낙관 A50/B30/C15/D05", {"A": .50, "B": .30, "C": .15, "D": .05}),
               ("기본 A25/B40/C28/D07", PRIORS),
               ("비관 A10/B30/C40/D20", {"A": .10, "B": .30, "C": .40, "D": .20})):
    best_s, best_v = None, -9e9
    for s in SCHEMES:
        ruin = sum(pr[m] * (store[(s, m)][0] < 0.5).mean() for m in pr)
        if ruin >= 0.10:
            continue
        el = sum(pr[m] * np.mean(np.log(np.maximum(store[(s, m)][0], 1e-6)))
                 for m in pr)
        if el > best_v:
            best_s, best_v = s, el
    print(f"  {nm:22s} → {best_s}  (기대 로그성장 {best_v:+.4f})")

# ---------------------------------------------------------------- kill switch
print()
print("=" * W)
print("3) 킬 스위치 — 언제 모델을 끌 것인가 (축차확률비 검정)")
print("=" * W)
print("가설 H1: 모델이 맞다(세계 A) / H0: 엣지 없다(세계 C)")
print("매 베팅마다 로그우도비를 누적하고, 임계값을 넘으면 결론을 낸다.\n")

p1 = pmo_a.mean()
p0 = pm_a.mean()
llr_win = np.log(p1 / p0)
llr_lose = np.log((1 - p1) / (1 - p0))
print(f"평균 모델확률 {p1:.3f} / 평균 시장확률 {p0:.3f}")
print(f"적중 1회당 증거 {llr_win:+.4f} / 실패 1회당 증거 {llr_lose:+.4f}")
A_up = np.log(19)      # 95% 확신으로 H1 채택
A_dn = np.log(1 / 19)  # 95% 확신으로 H0 채택 -> 모델 OFF
print(f"임계값: +{A_up:.2f} (엣지 확정) / {A_dn:.2f} (엣지 없음, 모델 OFF)\n")

print("실패가 연속될 때 몇 번째에 모델을 끄나:")
need_losses = int(np.ceil(A_dn / llr_lose))
print(f"  무승 연속 {need_losses}회 → 즉시 중단 (적중 0)")
print(f"{'베팅수':>6s} {'이 횟수에서 모델 OFF가 되는 최대 적중수':>40s} {'해당 적중률':>11s}")
print("-" * W)
for n in (20, 40, 60, 100, 150):
    best = None
    for k in range(0, n + 1):
        if k * llr_win + (n - k) * llr_lose <= A_dn:
            best = k
    if best is None:
        print(f"{n:6d} {'해당 없음':>40s} {'-':>11s}")
    else:
        print(f"{n:6d} {best:40d} {best/n:10.1%}")
print("-" * W)
print("읽는 법: 60베팅에서 적중이 이 숫자 이하면 '엣지 없음'이 95% 유력하다.")
print("그때는 파라미터를 미세조정하는 게 아니라 베팅을 멈추고 데이터를 다시 쌓는다.")

# ---------------------------------------------------------------- ev floor
print()
print("=" * W)
print("4) EV 컷의 함정 — 같은 EV가 배당에 따라 다른 난이도다")
print("=" * W)
print("EV 증가분 = 확률엣지(Δp) × 배당. 즉 같은 Δp가 긴 배당에서 더 큰 EV로 바뀐다.")
print("그래서 EV +4% 컷은 긴 배당에서 훨씬 느슨한 필터가 된다:\n")
print(f"{'배당':>6s} {'EV+4% 필요 Δp':>14s} {'레일(9%p) 꽉 채운 EV':>20s} {'Δp 4%p 컷 통과':>15s}")
print("-" * W)
for dec in (1.40, 1.60, 1.85, 2.10, 2.40, 2.80, 3.40):
    need_dp = 0.04 / dec
    pm = 1 / dec / 1.04
    ev_rail = min(0.97, pm + 0.09) * dec - 1
    print(f"{dec:6.2f} {need_dp*100:13.2f}%p {ev_rail:+19.1%} "
          f"{'통과' if 0.04 * dec >= 0.04 else '불가':>15s}")
print("-" * W)
print("단배 1.40에서 EV +4%를 만들려면 Δp 2.9%p가 필요한데, 단배 3.40에서는 1.2%p면 된다.")
print("내 모델 오차가 수 %p 단위이므로, 긴 배당에서는 EV컷이 노이즈를 걸러주지 못한다.")
print()
print("→ 따라서 EV컷 단독으로는 부족하고 '확률엣지 하한'을 같이 걸어야 한다.")
print("  규칙: Δp = |모델확률 - 시장확률| >= 4%p AND EV >= 4%(하키)/10%(축구)")
print(f"  오늘 보드 5베팅은 Δp가 모두 약 9%p(레일 한계)라 이 조건도 통과한다.")
print()
print("또 하나: EV = Δp × 배당 구조 때문에 긴 배당 쪽에서 EV가 크게 나온다.")
print("모델 픽이 역배로 몰리는 건 '역배 선호'가 아니라 이 수식의 결과다.")
print("동시에 그래서 역배 쪽을 더 엄격하게 봐야 한다 — EV 숫자가 쉽게 커지기 때문.")

print()
print("=" * W)
print("5) 채택 사이징을 오늘 보드에 적용")
print("=" * W)
CHOSEN = PICK[0]
f_ch = stake_fn(CHOSEN)
f_old = stake_fn("1/2켈리 상한4%")
print(f"채택: {CHOSEN}  (기존 권고 '1/2켈리 상한4%'는 가중 반토막 17.7%로 탈락)")
print(f"\n{'베팅':6s} {'배당':>5s} {'모델%':>6s} {'축소%':>6s} {'EV(모델)':>9s} "
      f"{'EV(축소)':>9s} {'기존 안':>8s} {'채택 안':>8s}")
print("-" * W)
tot_old = tot_new = 0.0
for i, (lb, pm, pmo, dec) in enumerate(BOARD):
    pshr = pm + 0.5 * (pmo - pm)
    s_old, s_new = f_old(i, dec), f_ch(i, dec)
    tot_old += s_old
    tot_new += s_new
    print(f"{lb:6s} {dec:5.2f} {pmo*100:5.1f}% {pshr*100:5.1f}% "
          f"{pmo*dec-1:+8.1%} {pshr*dec-1:+8.1%} {s_old*100:7.2f}% {s_new*100:7.2f}%")
print("-" * W)
DAY = 0.12
sc_old = min(1.0, DAY / tot_old)
sc_new = min(1.0, DAY / tot_new)
print(f"원시 총 스테이크  기존 {tot_old*100:.1f}% → 채택 {tot_new*100:.1f}%")
print(f"일일 상한 12% 적용 후 1베팅당  기존 {tot_old*sc_old/5*100:.2f}% → "
      f"채택 {tot_new*sc_new/5*100:.2f}%")
print()
print("핵심 변화: '엣지의 절반만 믿는다'를 먼저 적용하고 그 위에 하프켈리를 쓴다.")
print("오늘은 일일 상한이 먼저 걸려서 금액 자체는 비슷하지만, 엣지가 작은 날에는")
print("채택안이 훨씬 작게 베팅하고 그게 세계 C/D에서 생존을 만든다.")

print()
print("-" * W)
print("일관성 점검 — 선별도 축소확률로 해야 한다")
print("-" * W)
print("선별은 낙관적 확률(모델)로 하고 금액만 보수적 확률(축소)로 쓰면 모순이다.")
print("확률 추정치를 하나로 정하고 선별·금액에 같이 적용해야 한다.")
print(f"\n축소확률 기준으로 컷을 다시 걸면:")
print(f"{'베팅':6s} {'종목':5s} {'EV(축소)':>9s} {'적용 컷':>8s} {'판정':>8s}")
print("-" * W)
FLOORS = {"SEA": ("하키", 0.04), "NSH": ("하키", 0.04), "PHI": ("하키", 0.04),
          "ENG": ("축구", 0.10), "SCO": ("축구", 0.10)}
survivors = []
for lb, pm, pmo, dec in BOARD:
    pshr = pm + 0.5 * (pmo - pm)
    ev = pshr * dec - 1
    sport, fl = FLOORS[lb]
    ok = ev >= fl
    if ok:
        survivors.append(lb)
    print(f"{lb:6s} {sport:5s} {ev:+8.1%} {fl:+7.0%} {'통과' if ok else '탈락':>8s}")
print("-" * W)
print(f"생존: {', '.join(survivors)}  ({len(survivors)}/5)")
print("축구 두 개는 축소확률로 보면 EV +3.8%/+2.7%로, 축구 컷 +10%는 물론")
print("하키 컷 +4%도 못 넘는다. 즉 보수적 추정으로 가면 오늘 축구는 전부 PASS다.")
print()
print("이게 '엣지를 절반만 믿기'의 실제 비용이다. 베팅 수가 5개에서 3개로 줄고")
print("하루 기대수익도 줄지만, 세계 C/D에서 살아남는 대가로 지불하는 값이다.")
print("p=0.080 상태에서는 이쪽이 맞다. 엣지가 누적 데이터로 확인되면 축소율을 낮춘다.")

print()
print("=" * W)
print("6) 최종 전략")
print("=" * W)
print("""
[자금]
  뱅크롤 전액 분리. 생활비와 섞으면 아래 계산 전부가 무의미해진다.
  사이징 = 엣지를 절반으로 축소한 뒤 하프켈리, 1베팅 상한 4%, 하루 총 12%.
  풀켈리 금지. 엣지가 미확정(p=0.080)이라 세계 D에서 반토막 확률 100%.
  모델 엣지를 그대로 쓴 하프켈리도 금지 — 가중 반토막 17.7%로 기준 초과.

[선별]
  1. 시장 종가 무공제 확률에서 출발. 내 의견은 여기서 시작하지 않는다.
  2. 전역항(페이브세 -0.103, 원정 페이브 -0.050). 시즌 기저율이라 신뢰도 높음.
  3. 상황태그(결장·골리·B2B·노나먹기). 부호만 믿고 크기는 사전값 유지.
  4. ±9%p 레일. 레일에 닿으면 '모델이 과신 중'이라는 신호로 읽는다.
  5. 이중 하한: Δp >= 4%p AND EV >= 4%(하키)/10%(축구). 둘 다 넘어야 베팅.
     EV컷 단독은 긴 배당에서 느슨해진다(EV = Δp × 배당).
  6. MLB 전면 제외 — 백테스트 없음.

[실행]
  라인업 확정 후에만 집행. 태그 2개가 깨지면 취소.
  단폴만. 조합은 공제가 곱해져 EV가 사라진다.
  종가 근처에서 집행. 종가가 모델 입력이므로 그래야 계산이 유효하다.

[기록]
  매 베팅: 날짜·경기·대상·배당·시장확률·모델확률·Δp·EV·스테이크·떴던 태그·결과.
  태그별 누적이 있어야 재적합이 된다. 결과만 적으면 데이터가 아니다.

[점검]
  주 단위 재적합. 가중치는 주당 사전값 대비 소폭까지만 이동.
  킬 스위치: 20베팅 1적중 이하 / 40베팅 11적중 이하 / 60베팅 21적중 이하
            / 100베팅 40적중 이하 → 즉시 중단, 파라미터 미세조정 금지, 데이터 재수집.
  무승 17연패도 즉시 중단.
  전역항은 월 단위 재측정. 57.3%가 그 시즌 특수일 수 있다.

[금지]
  확률만 높은 픽(ESP·COL·BUF 유형) — 100베팅 손실확률 50.7%.
  무조건 역배 / 무조건 +1.5 — 지난주 +30.7%는 그 주가 역배장이었던 것.
  손실 복구용 증액. 켈리는 뱅크롤 비례라 자동으로 줄어든다. 그게 맞는 거다.
  단일 경기로 파라미터 수정.

[기대치]
  엣지가 모델대로라면 150베팅 중위 1.55배.
  엣지가 절반이면 1.14배. 엣지가 없으면 0.83배.
  베이즈 가중 기대 로그성장 +0.051. 하루 +1.8%는 '좋은 날'의 숫자이고
  장기 중위는 이보다 훨씬 완만하다. 이걸 기대치로 잡아야 중간에 안 흔들린다.
""".strip())
