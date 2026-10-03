"""
Apply the optimized model to today's KST board (Oct 4 KST = Oct 3 ET/CET)
and stress-test it.

The point of this file, beyond repeating EV numbers: every NHL edge here comes
from feature tags I assigned by hand (who is out, who is on a backup goalie,
who is on the second of a back-to-back). If a tag is wrong the edge can vanish.
So each bet is decomposed into

    market price  ->  global season terms  ->  each individual tag

and then re-priced with tags removed one at a time, so it is visible how much
of the bet rests on information that could still change before puck drop.
"""
import numpy as np
from nhl_backtest import am_to_dec, logit, sigmoid, novig

W_NHL = np.load("/workspace/W_v4.npy")
W_SOC = np.load("/workspace/W_soccer_v4.npy")
B0, B_AWAY = -0.1031, -0.050
RAIL, EV_FLOOR, KCAP, HALF, DAY_CAP = 0.09, 0.04, 0.04, 0.5, 0.12
CODES = ["fs", "ds", "fg", "dg", "fb", "db", "rl", "up"]
WMAP = dict(zip(CODES, W_NHL))
TAGNAME = {
    "fs": "페이브 주력 결장", "ds": "역배 주력 결장",
    "fg": "페이브 골리 다운그레이드", "dg": "역배 골리 다운그레이드",
    "fb": "페이브 B2B 뒷경기", "db": "역배 B2B 뒷경기",
    "rl": "노나먹기", "up": "역배 보강 우위",
}

# matchup is AWAY@HOME. (matchup, fav, fav_ml, dog_ml, feats, note)
NHL = [
    ("CHI@BUF", "BUF", -230, +190, dict(ds=2), "시카고 주력 2명 결장"),
    ("MTL@PIT", "MTL", -125, +105, dict(), "특이사항 없음"),
    ("WSH@TB",  "TB",  -155, +130, dict(db=1, dg=1), "WSH B2B + 백업골리"),
    ("OTT@TOR", "TOR", -130, +110, dict(), "특이사항 없음"),
    ("CAR@PHI", "CAR", -130, +110, dict(fs=1, fg=1, fb=1), "자비스 OUT + 크리스 + B2B"),
    ("UTA@CBJ", "UTA", -112, -108, dict(), "실질 픽엠"),
    ("SEA@EDM", "EDM", -205, +170, dict(fs=2, fg=1), "EDM 주력 2명 + 골리"),
    ("NJ@NYI",  "NJ",  -130, +110, dict(ds=1), "NYI 주력 1명 결장"),
    ("BOS@MIN", "MIN", -180, +150, dict(ds=1, db=1), "BOS 주력 결장 + B2B"),
    ("DAL@NSH", "DAL", -142, +120, dict(fs=1, fb=1), "DAL 주력 1명 + B2B"),
    ("STL@COL", "COL", -298, +240, dict(db=1), "STL B2B / COL 초강세 가격"),
    ("CGY@VAN", "VAN", -110, -110, dict(), "픽엠"),
    ("LAK@SJS", "SJS", -120, +100, dict(ds=1), "LAK 주력 1명 결장"),
]


def dog_of(matchup, fav):
    away, home = matchup.split("@")
    return away if fav == home else home


def nhl_p(fav_ml, dog_ml, fav_home, feats):
    pm = novig(fav_ml, dog_ml)
    x = np.array([feats.get(c, 0) for c in CODES], float)
    g = B0 + B_AWAY * (1 - fav_home)
    p = sigmoid(logit(pm) + g + x @ W_NHL)
    return pm, float(np.clip(p, max(.01, pm - RAIL), min(.99, pm + RAIL)))


def kelly(p, dec):
    b = dec - 1
    return min(KCAP, max(0.0, (p * b - (1 - p)) / b) * HALF)


def evaluate(matchup, fav, fml, dml, feats):
    away, home = matchup.split("@")
    fav_home = 1 if fav == home else 0
    pm, p = nhl_p(fml, dml, fav_home, feats)
    fd, dd = am_to_dec(fml), am_to_dec(dml)
    ev_f, ev_d = p * fd - 1, (1 - p) * dd - 1
    if ev_f >= ev_d:
        return pm, p, fav, fd, ev_f, "FAV"
    return pm, p, dog_of(matchup, fav), dd, ev_d, "DOG"


print("=" * 100)
print("한국시간 10/4 보드 — 최적화 모델 적용")
print("(= ET/CET 10/3. NHL 정규리그 13경기 + UEFA NL 4경기 + MLB DS 1차전)")
print("=" * 100)

rows = []
for mu, fav, fml, dml, feats, note in NHL:
    pm, p, side, dec, ev, which = evaluate(mu, fav, fml, dml, feats)
    stake = kelly(p if which == "FAV" else 1 - p, dec) if ev >= EV_FLOOR else 0.0
    val = max(0.0, stake * ev * 10000)
    rows.append(dict(mu=mu, fav=fav, fml=fml, dml=dml, feats=feats, note=note,
                     pm=pm, p=p, side=side, dec=dec, ev=ev, which=which,
                     stake=stake, val=val))

print()
print("1) NHL 13경기 전수 — 어디에 가치가 있고 어디에 없나")
print("-" * 100)
print(f"{'경기':10s} {'페이브':5s} {'시장%':>6s} {'모델%':>6s} {'추천':5s} {'배당':>5s} "
      f"{'EV':>7s} {'스테이크':>7s} {'가치':>7s}  비고")
for r in sorted(rows, key=lambda r: -r["val"]):
    mark = "" if r["val"] > 0 else "   ← 가치 0"
    print(f"{r['mu']:10s} {r['fav']:5s} {r['pm']*100:5.1f}% {r['p']*100:5.1f}% "
          f"{r['side']:5s} {r['dec']:5.2f} {r['ev']:+6.1%} {r['stake']*100:6.1f}% "
          f"{r['val']:6.0f}bp  {r['note']}{mark}")

live = [r for r in rows if r["val"] > 0]
print("-" * 100)
print(f"가치 있는 경기 {len(live)}/13. 나머지 10경기는 EV컷 +4% 미달 → 전부 PASS.")

# ---------------------------------------------------------------- decomposition
print()
print("=" * 100)
print("2) 가치 분해 — 이 EV가 어디서 나오는가")
print("=" * 100)
print("시장가격에서 출발해 전역항, 그 다음 태그를 하나씩 더해가며 추천측 확률 변화")
for r in live:
    away, home = r["mu"].split("@")
    fav_home = 1 if r["fav"] == home else 0
    print(f"\n[{r['mu']}] 추천 {r['side']} @ {r['dec']:.2f}  (최종 EV {r['ev']:+.1%})")
    pm = r["pm"]
    cur = logit(pm)
    side_is_dog = r["which"] == "DOG"

    def show(label, z):
        p_fav = sigmoid(z)
        p_side = 1 - p_fav if side_is_dog else p_fav
        ev = p_side * r["dec"] - 1
        print(f"   {label:34s} 추천측 {p_side*100:5.1f}%   EV {ev:+6.1%}")

    show("시장 종가(무공제)", cur)
    g = B0 + B_AWAY * (1 - fav_home)
    cur += g
    show(f"+ 전역항 (페이브세{'·원정' if not fav_home else ''}) {g:+.3f}", cur)
    for code, v in r["feats"].items():
        cur += WMAP[code] * v
        label = f"+ {TAGNAME[code]}" + (f" x{v}" if v > 1 else "")
        show(f"{label} ({WMAP[code]*v:+.3f})", cur)
    p_raw = sigmoid(cur)
    p_raw_side = 1 - p_raw if side_is_dog else p_raw
    clipped = abs(p_raw - np.clip(p_raw, pm - RAIL, pm + RAIL)) > 1e-9
    if clipped:
        print(f"   {'= 레일 ±9% 적용 (여기서 잘림)':34s} 추천측 "
              f"{(1-r['p'] if side_is_dog else r['p'])*100:5.1f}%   EV {r['ev']:+6.1%}")
        print(f"     레일 없으면 {p_raw_side*100:.1f}% / EV {p_raw_side*r['dec']-1:+.1%} "
              f"였다. 레일이 {p_raw_side*r['dec']-1 - r['ev']:+.1%}p 깎아냈다.")
    else:
        print(f"   {'= 레일 미작동 (한도 내)':34s}")

# ---------------------------------------------------------------- stress test
print()
print("=" * 100)
print("3) 태그 민감도 — 태그가 틀리면 베팅이 살아남는가")
print("=" * 100)
from itertools import combinations

print("태그가 사실과 다를 경우(복귀·골리 변경) 최악 시나리오별 EV")
print(f"{'경기':10s} {'추천':5s} {'전체':>7s} {'1개 깨짐':>9s} {'2개 깨짐':>9s} "
      f"{'전부 깨짐':>9s}  몇 개까지 버티나")
print("-" * 100)
for r in live:
    away, home = r["mu"].split("@")
    fav_home = 1 if r["fav"] == home else 0
    codes = list(r["feats"])

    def ev_with(f2):
        _, p2 = nhl_p(r["fml"], r["dml"], fav_home, f2)
        p_side = (1 - p2) if r["which"] == "DOG" else p2
        return p_side * r["dec"] - 1

    worst = {}
    for k in range(0, len(codes) + 1):
        evs = [ev_with({c: v for c, v in r["feats"].items() if c not in drop})
               for drop in combinations(codes, k)]
        worst[k] = min(evs)
    tol = max([k for k in worst if worst[k] >= EV_FLOOR], default=-1)
    c1 = f"{worst[1]:+.1%}" if 1 in worst and len(codes) >= 1 else "-"
    c2 = f"{worst[2]:+.1%}" if len(codes) >= 2 else "-"
    cn = f"{worst[len(codes)]:+.1%}"
    if tol >= len(codes):
        msg = ("전부 깨져도 유효(단 컷 경계선)"
               if worst[len(codes)] < EV_FLOOR + 0.01 else "전부 깨져도 유효")
    elif tol >= 1:
        msg = f"{tol}개까지 버팀"
    else:
        msg = "하나라도 깨지면 PASS"
    print(f"{r['mu']:10s} {r['side']:5s} {r['ev']:+6.1%} {c1:>9s} {c2:>9s} "
          f"{cn:>9s}  {msg}")

print()
print("읽는 법: 이 베팅들은 '배당이 싸다'가 아니라 '라인업 정보가 아직 가격에")
print("안 들어갔다'는 베팅이다. SEA는 태그를 전부 빼면 +2.4%로 컷 아래로 떨어진다")
print("— 즉 EDM 결장자가 복귀하면 베팅 근거 자체가 사라진다.")
print("반대로 PHI·NSH는 ±9% 레일이 이미 EV를 깎아놨기 때문에 태그 한두 개가")
print("틀려도 컷을 넘는다. 레일이 과신 방지 장치로 실제 작동하고 있다는 뜻이다.")
print("실행 전 반드시 확인: 선발 골리 발표, 결장자 복귀 여부, B2B 일정.")

# ---------------------------------------------------------------- soccer
print()
print("=" * 100)
print("4) UEFA NL 4경기")
print("=" * 100)
OVR, RAIL_S, EV_FLOOR_S = 1.06, 0.20, 0.10
SOC = [
    ("CRO-ENG", "ENG", 2.10, 1, 0, "잉글랜드 체급 우위 / 원정"),
    ("ESP-CZE", "ESP", 1.25, 1, 1, "스페인 압도적이지만 가격에 이미 반영"),
    ("SUI-SVN", "SUI", 1.70, 1, 1, "스위스 우위, 티어갭 판정 애매"),
    ("MKD-SCO", "SCO", 1.85, 1, 0, "스코틀랜드 체급 우위 / 원정"),
]
print(f"{'경기':10s} {'페이브':5s} {'시장%':>6s} {'모델%':>6s} {'배당':>5s} {'EV':>7s} "
      f"{'스테이크':>7s} {'가치':>7s}  비고")
srows = []
for mt, fav, fd, tg, fh, note in SOC:
    pm = min(max((1 / fd) / OVR, .02), .95)
    z = np.array([tg, 1 - tg, 1 - fh], float)
    p = float(np.clip(sigmoid(logit(pm) + z @ W_SOC),
                      max(.01, pm - RAIL_S), min(.97, pm + RAIL_S)))
    ev = p * fd - 1
    st = kelly(p, fd) if ev >= EV_FLOOR_S else 0.0
    val = max(0.0, st * ev * 10000)
    srows.append(dict(mt=mt, fav=fav, pm=pm, p=p, dec=fd, ev=ev, stake=st,
                      val=val, tg=tg, fh=fh, note=note))
for r in sorted(srows, key=lambda r: -r["val"]):
    mark = "" if r["val"] > 0 else "   ← 가치 0 (EV컷 +10% 미달)"
    print(f"{r['mt']:10s} {r['fav']:5s} {r['pm']*100:5.1f}% {r['p']*100:5.1f}% "
          f"{r['dec']:5.2f} {r['ev']:+6.1%} {r['stake']*100:6.1f}% {r['val']:6.0f}bp  "
          f"{r['note']}{mark}")

print("\n티어갭 민감도 — 티어갭이 없다고 보면(S3 적용) 어떻게 되나:")
for r in srows:
    z0 = np.array([0, 1, 1 - r["fh"]], float)
    p0 = float(np.clip(sigmoid(logit(r["pm"]) + z0 @ W_SOC),
                       max(.01, r["pm"] - RAIL_S), min(.97, r["pm"] + RAIL_S)))
    print(f"  {r['mt']:10s} {r['fav']:4s} EV {r['ev']:+6.1%} → {p0*r['dec']-1:+6.1%}"
          f"  ({'여전히 유효' if p0*r['dec']-1 >= EV_FLOOR_S else '베팅 불가'})")
print("\n축구 모델은 티어갭 판정 하나에 전부 걸려 있다. 애매하면 PASS가 기본값.")

# ---------------------------------------------------------------- mlb
print()
print("=" * 100)
print("5) MLB 디비전시리즈 1차전 — 가치 산정 불가")
print("=" * 100)
for t, ml in [("CLE", -157), ("LAD", -206), ("TB", -141), ("MIL", -225)]:
    d = am_to_dec(ml)
    print(f"  {t:4s} {ml:5d} (배당 {d:.2f}, 공제전 암시 {1/d:.1%})  →  모델 없음, PASS")
print("\n백테스트를 하키 정규리그와 축구만 돌렸다. MLB는 검증된 가중치가 없어서")
print("EV를 계산할 근거가 없다. 가치 0으로 두는 게 정직하고, 정규시즌 파라미터를")
print("빌려오는 것도 안 된다(포스트시즌은 불펜·로테이션 운용이 다른 경기다).")

# ---------------------------------------------------------------- final
print()
print("=" * 100)
print("6) 오늘 최종 보드")
print("=" * 100)
allr = ([(r["mu"], "NHL", r["side"], r["dec"], r["ev"], r["stake"], r["val"],
          r["p"] if r["which"] == "FAV" else 1 - r["p"]) for r in live]
        + [(r["mt"], "축구", r["fav"], r["dec"], r["ev"], r["stake"], r["val"], r["p"])
           for r in srows if r["val"] > 0])
allr.sort(key=lambda x: -x[6])
raw = sum(x[5] for x in allr)
sc = min(1.0, DAY_CAP / raw) if raw else 1.0
T = sum(x[6] for x in allr) * sc

print(f"{'구분':6s} {'경기':10s} {'대상':5s} {'배당':>5s} {'모델확률':>8s} {'EV':>7s} "
      f"{'스테이크':>7s} {'가치':>7s}")
print("-" * 100)
for i, (mu, sp, side, dec, ev, f, v, pside) in enumerate(allr):
    tier = "주력" if ev >= 0.14 else "부주력"
    print(f"{tier:6s} {mu:10s} {side:5s} {dec:5.2f} {pside*100:7.1f}% {ev:+6.1%} "
          f"{f*sc*100:6.2f}% {v*sc:6.0f}bp")
print("-" * 100)
print(f"총 {len(allr)}베팅 / 총 스테이크 {raw*sc*100:.1f}% (일일 상한 {DAY_CAP:.0%}) / "
      f"기대 뱅크롤 성장 {T:.0f}bp = +{T/100:.2f}%")
anchor = max(allr, key=lambda x: x[7])
print(f"축(조합 기준) : {anchor[2]} — 모델확률 {anchor[7]*100:.1f}%, 보드 내 최고")
print("단 조합은 공제가 곱해져 위 EV가 날아간다. 숫자상 전부 단폴이 맞다.")
print(f"\nPASS: NHL 10경기 + ESP·SUI + MLB 4경기. 오늘 보드의 2/3는 가치가 없다.")
