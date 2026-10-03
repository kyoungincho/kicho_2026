"""
Forward betting-value calculator (KST Oct 4 board).

Answers one question per game, numerically: how much is this bet actually worth?

Value metric
------------
  EV      = p_model * (dec - 1) - (1 - p_model)      per 1 unit staked
  f       = half-Kelly fraction of bankroll, capped at 4%
  VALUE   = f * EV, in basis points of bankroll
            = expected bankroll growth from taking this bet at its proper size

VALUE is the honest ranking number. A bet with a juicy EV but a price so short
that Kelly only lets you put 0.4% down is worth less than a modest EV at a
size you can actually take. A bet below the EV floor scores 0 - no value,
regardless of how confident the write-up sounded.

Hockey model: v4 (regular-season data only).
Soccer model: soccer v4.
Baseball:     no backtested model exists -> no value number is produced.
"""
import numpy as np
from nhl_backtest import am_to_dec, logit, sigmoid, novig

# ------------------------------------------------------------------ NHL
W_NHL = np.load("/workspace/W_v4.npy")
B0, B_AWAY = -0.1031, -0.050
RAIL, EV_FLOOR, KCAP, HALF = 0.09, 0.04, 0.04, 0.5
CODES = ["fs", "ds", "fg", "dg", "fb", "db", "rl", "up"]

# (matchup, fav, fav_ml, dog_ml, fav_home, feats, note)
NHL = [
    ("CHI@BUF", "BUF", -230, +190, 1, dict(ds=2),
     "시카고 주력 2명 결장 / 버팔로 홈"),
    ("MTL@PIT", "MTL", -125, +105, 0, dict(),
     "특이사항 없음"),
    ("WSH@TB",  "TB",  -155, +130, 1, dict(db=1, dg=1),
     "워싱턴 B2B 뒷경기 + 골리 백업"),
    ("OTT@TOR", "TOR", -130, +110, 1, dict(),
     "특이사항 없음"),
    ("CAR@PHI", "CAR", -130, +110, 0, dict(fs=1, fg=1, fb=1),
     "캐롤라이나 자비스 OUT + 크리스 불안 + B2B"),
    ("UTA@CBJ", "UTA", -112, -108, 0, dict(),
     "실질 픽엠"),
    ("SEA@EDM", "EDM", -205, +170, 1, dict(fs=2, fg=1),
     "에드먼턴 주력 2명 + 골리 다운그레이드"),
    ("NJ@NYI",  "NJ",  -130, +110, 0, dict(ds=1),
     "아일런더스 주력 1명 결장"),
    ("BOS@MIN", "MIN", -180, +150, 1, dict(ds=1, db=1),
     "보스턴 주력 결장 + B2B 뒷경기"),
    ("DAL@NSH", "DAL", -142, +120, 0, dict(fs=1, fb=1),
     "댈러스 주력 1명 + B2B"),
    ("STL@COL", "COL", -298, +240, 1, dict(db=1),
     "세인트루이스 B2B / 콜로라도 초강세 가격"),
    ("CGY@VAN", "VAN", -110, -110, 1, dict(),
     "픽엠"),
    ("LAK@SJS", "SJS", -120, +100, 1, dict(ds=1),
     "LA 주력 1명 결장"),
]


def nhl_predict(fav_ml, dog_ml, fav_home, feats):
    pm = novig(fav_ml, dog_ml)
    x = np.array([feats.get(c, 0) for c in CODES], float)
    g = B0 + B_AWAY * (1 - fav_home)
    p = sigmoid(logit(pm) + g + x @ W_NHL)
    p = float(np.clip(p, max(.01, pm - RAIL), min(.99, pm + RAIL)))
    return pm, p


def kelly(p, dec, cap=KCAP, half=HALF):
    b = dec - 1
    f = (p * b - (1 - p)) / b
    return min(cap, max(0.0, f) * half)


rows = []
for mu, fav, fml, dml, fh, feats, note in NHL:
    pm, p = nhl_predict(fml, dml, fh, feats)
    fd, dd = am_to_dec(fml), am_to_dec(dml)
    ev_f = p * fd - 1
    ev_d = (1 - p) * dd - 1
    if ev_f >= ev_d:
        side, pp, dec, ev = fav, p, fd, ev_f
    else:
        dog = mu.split("@")[0] if fav == mu.split("@")[1] else mu.split("@")[1]
        side, pp, dec, ev = dog, 1 - p, dd, ev_d
    f = kelly(pp, dec) if ev >= EV_FLOOR else 0.0
    val = max(0.0, f * ev * 10000)
    rows.append((mu, fav, pm, p, side, dec, ev, f, val, note))

print("=" * 108)
print("1) NHL (KST 10/4 · 정규리그 v4 모델) — 배팅 가치 숫자화")
print("=" * 108)
print(f"{'경기':10s} {'페이브':5s} {'시장%':>6s} {'모델%':>6s} {'추천':5s} "
      f"{'배당':>5s} {'EV':>7s} {'스테이크':>7s} {'가치(bp)':>8s}  비고")
print("-" * 108)
for mu, fav, pm, p, side, dec, ev, f, val, note in sorted(rows, key=lambda r: -r[8]):
    flag = "" if val > 0 else "  ← 가치없음"
    print(f"{mu:10s} {fav:5s} {pm*100:5.1f}% {p*100:5.1f}% {side:5s} "
          f"{dec:5.2f} {ev:+6.1%} {f*100:6.1f}% {val:8.0f}  {note}{flag}")

tot = sum(r[8] for r in rows)
act = [r for r in rows if r[8] > 0]
print("-" * 108)
print(f"베팅 가치 있는 경기 {len(act)}/{len(rows)}개, 총 기대 뱅크롤 성장 {tot:.0f}bp "
      f"({tot/100:.2f}%)")

# ------------------------------------------------------------------ Soccer
W_S = np.load("/workspace/W_soccer_v4.npy")
OVR, RAIL_S, EV_FLOOR_S = 1.06, 0.20, 0.10
P_DRAW_NOGAP = 0.320

# (match, fav, fav_dec, tier_gap, fav_home, draw_dec, note)
SOC = [
    ("CRO-ENG", "ENG", 2.10, 1, 0, 3.30, "잉글랜드 체급 우위 / 원정"),
    ("ESP-CZE", "ESP", 1.25, 1, 1, 5.75, "스페인 압도적 체급차"),
    ("SUI-SVN", "SUI", 1.70, 1, 1, 3.60, "스위스 우위지만 체급차 판정 애매"),
    ("MKD-SCO", "SCO", 1.85, 1, 0, 3.45, "스코틀랜드 체급 우위 / 원정"),
]

print()
print("=" * 108)
print("2) 축구 UEFA NL (KST 10/4 · soccer v4) — 배팅 가치 숫자화")
print("=" * 108)
print(f"{'경기':10s} {'페이브':5s} {'시장%':>6s} {'모델%':>6s} {'배당':>5s} "
      f"{'EV':>7s} {'스테이크':>7s} {'가치(bp)':>8s}  비고")
print("-" * 108)
srows = []
for mt, fav, fd, tg, fh, dd_draw, note in SOC:
    pm = min(max((1 / fd) / OVR, .02), .95)
    z = np.array([tg, 1 - tg, 1 - fh], float)
    p = sigmoid(logit(pm) + z @ W_S)
    p = float(np.clip(p, max(.01, pm - RAIL_S), min(.97, pm + RAIL_S)))
    ev = p * fd - 1
    f = kelly(p, fd) if ev >= EV_FLOOR_S else 0.0
    val = f * ev * 10000
    srows.append((mt, fav, pm, p, fd, ev, f, val, note, tg))
for mt, fav, pm, p, fd, ev, f, val, note, tg in sorted(srows, key=lambda r: -r[7]):
    flag = "" if val > 0 else "  ← 가치없음(EV컷 +10% 미달)"
    print(f"{mt:10s} {fav:5s} {pm*100:5.1f}% {p*100:5.1f}% {fd:5.2f} "
          f"{ev:+6.1%} {f*100:6.1f}% {val:8.0f}  {note}{flag}")
tot_s = sum(r[7] for r in srows)
print("-" * 108)
print(f"총 기대 뱅크롤 성장 {tot_s:.0f}bp ({tot_s/100:.2f}%)")
print(f"\n티어갭 판정이 모델의 전부다. 애매하면(S3) 페이브 가치는 즉시 음수로 바뀐다.")
print(f"티어갭 없다고 보면 무승부 p={P_DRAW_NOGAP:.0%} → 단배 3.2+ 에서만 소액.")

# ------------------------------------------------------------------ MLB
print()
print("=" * 108)
print("3) MLB 디비전시리즈 1차전 — 모델 없음")
print("=" * 108)
MLB = [("CLE", -157), ("LAD", -206), ("TB", -141), ("MIL", -225)]
print(f"{'페이브':6s} {'ML':>6s} {'배당':>5s} {'시장 암시(공제전)':>16s} {'가치(bp)':>9s}  판정")
print("-" * 108)
for t, ml in MLB:
    d = am_to_dec(ml)
    print(f"{t:6s} {ml:6d} {d:5.2f} {1/d:15.1%} {0:9d}  모델 미검증 → PASS")
print("-" * 108)
print("MLB는 이번 최적화에 포함된 백테스트가 없다(하키 정규리그·축구만).")
print("검증된 가중치가 없으면 EV를 계산할 근거가 없으므로 가치는 0으로 둔다.")
print("포스트시즌은 불펜 전면가동·로테이션 단축으로 정규시즌 파라미터도 전이되지 않는다.")

# ------------------------------------------------------------------ verdict
print()
print("=" * 108)
print("4) 지난 보드(KST 10/4 예고분) 재평가 — 최적화 후에도 가치가 남았나")
print("=" * 108)
prev = {
    "SEA": ("NHL", "SEA@EDM 역배", 8),
    "TB":  ("NHL", "TB ML", 5),
    "ESP": ("SOC", "ESP ML", 5),
    "SUI": ("SOC", "SUI ML", 4),
    "MIL": ("MLB", "MIL ML", 4),
}
lookup_n = {r[0]: r for r in rows}
print(f"{'기존픽':6s} {'기존점수':>7s} {'최적화 후 가치':>14s}  결론")
print("-" * 108)
for k, (sp, label, sc) in prev.items():
    if sp == "NHL":
        rr = [r for r in rows if k in r[0] or k == r[4]]
        v = rr[0][8] if rr else 0.0
        ev = rr[0][6] if rr else 0.0
        keep = (rr and rr[0][4] == k and v > 0)
        msg = (f"유지 (EV {ev:+.1%})" if keep
               else "추천측이 바뀜/컷 미달 → 가치 소멸")
    elif sp == "SOC":
        rr = [r for r in srows if r[1] == k]
        v = rr[0][7] if rr else 0.0
        ev = rr[0][5] if rr else 0.0
        msg = f"유지 (EV {ev:+.1%})" if v > 0 else "EV컷 미달 → 가치 소멸"
    else:
        v = 0.0
        msg = "모델 없음 → 가치 산정 불가, 철회"
    print(f"{label:6s} {('+'+str(sc)):>7s} {v:13.0f}bp  {msg}")

print("\n반대 방향 — 기존에 PASS였는데 최적화 후 가치가 생긴 경기:")
prev_pass = {"CAR@PHI", "CRO-ENG", "STL@COL", "CHI@BUF", "DAL@NSH"}
flipped = ([(r[0], r[4], r[6], r[8]) for r in rows if r[0] in prev_pass and r[8] > 0]
           + [(r[0], r[1], r[5], r[7]) for r in srows if r[0] in prev_pass and r[7] > 0])
for mu, side, ev, v in sorted(flipped, key=lambda r: -r[3]):
    print(f"  {mu:10s} → {side:5s} EV {ev:+6.1%}  가치 {v:.0f}bp  (기존 PASS 판정이 틀렸다)")
if not flipped:
    print("  없음")

print()
print("=" * 108)
print("5) 최종 — 가치 순위 (전 종목 통합, bp = 뱅크롤 기대성장)")
print("=" * 108)
allr = ([(r[0], "NHL", r[4], r[5], r[6], r[7], r[8]) for r in rows if r[8] > 0]
        + [(r[0], "축구", r[1], r[4], r[5], r[6], r[7]) for r in srows if r[7] > 0])
allr.sort(key=lambda r: -r[6])
if not allr:
    print("가치 있는 베팅 없음 → 전면 PASS")
else:
    print(f"{'#':>2s} {'경기':10s} {'종목':5s} {'대상':5s} {'배당':>5s} {'EV':>7s} "
          f"{'스테이크':>7s} {'가치':>8s}")
    print("-" * 108)
    for i, (mu, sp, side, dec, ev, f, v) in enumerate(allr, 1):
        print(f"{i:2d} {mu:10s} {sp:5s} {side:5s} {dec:5.2f} {ev:+6.1%} "
              f"{f*100:6.1f}% {v:7.0f}bp")
    T = sum(r[6] for r in allr)
    raw_exp = sum(r[5] for r in allr)
    print("-" * 108)
    print(f"합계 {len(allr)}베팅 / 원시 총 스테이크 {raw_exp*100:.1f}% / "
          f"기대 뱅크롤 성장 {T:.0f}bp (+{T/100:.2f}%)")

    # Daily exposure cap. Five independent 4% bets is 20% of bankroll at risk in
    # one night; the model's edge is not certain enough to justify that. Scale
    # the whole slate down to the cap, which scales expected growth with it.
    DAY_CAP = 0.12
    print()
    print("=" * 108)
    print(f"6) 일일 노출 상한 적용 (하루 총 스테이크 {DAY_CAP:.0%})")
    print("=" * 108)
    sc = min(1.0, DAY_CAP / raw_exp) if raw_exp else 1.0
    print(f"{'#':>2s} {'경기':10s} {'대상':5s} {'조정 스테이크':>12s} {'가치':>9s}")
    print("-" * 108)
    for i, (mu, sp, side, dec, ev, f, v) in enumerate(allr, 1):
        print(f"{i:2d} {mu:10s} {side:5s} {f*sc*100:11.2f}% {v*sc:8.0f}bp")
    print("-" * 108)
    print(f"조정 후: 총 스테이크 {raw_exp*sc*100:.1f}% / "
          f"기대 뱅크롤 성장 {T*sc:.0f}bp (+{T*sc/100:.2f}%)")
    print(f"\n해석: 제대로 사이징하면 이 보드 하루 전체의 기댓값은 뱅크롤 +{T*sc/100:.2f}%다.")
    print("      '오늘의 베스트'라는 표현이 무의미한 이유 — 하루 정당한 기댓값은 이 크기고,")
    print("      가치 0으로 찍힌 경기에 돈을 넣는 순간 이 숫자는 바로 마이너스로 간다.")
    print(f"\n참고: 위 EV는 '내가 붙인 피처 태그가 맞다'는 가정 하의 값이다. 라인업이")
    print("      바뀌면(골리 발표·결장 복귀) EV는 재계산해야 하고 부호도 바뀔 수 있다.")
