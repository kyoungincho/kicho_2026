"""
UEFA Nations League 2026-27, matchday 1-3 (Sep 25 - Oct 2, 2026).
Purpose: test the soccer-side parameters (S1..S7) that were written from
memory, against what actually happened.

Soccer is 1X2, so a "favorite" can fail by drawing. That makes short home
favorites structurally worse than the same price on a hockey moneyline.
"""
import numpy as np

# (match, fav_side, fav_dec_approx, tier_gap, home_fav, result_for_fav)
# tier_gap = 1 if the favorite is a clearly higher-tier nation (not just home adv)
# result: 'W' fav won, 'D' draw, 'L' fav lost
M = [
    ("ITA 0-2 BEL",  "ITA", 2.35, 0, 1, "L"),
    ("TUR 0-1 FRA",  "FRA", 1.50, 1, 0, "W"),
    ("HUN 0-1 UKR",  "HUN", 2.40, 0, 1, "L"),
    ("GEO 0-1 NIR",  "GEO", 2.10, 0, 1, "L"),
    ("POL 0-0 BIH",  "POL", 1.70, 0, 1, "D"),
    ("SWE 2-1 ROU",  "SWE", 1.55, 1, 1, "W"),
    ("NED 1-1 GER",  "NED", 2.20, 0, 1, "D"),
    ("SRB 1-2 GRE",  "GRE", 2.40, 1, 0, "W"),
    ("NOR 3-2 DEN",  "NOR", 1.90, 0, 1, "W"),
    ("POR 1-0 WAL",  "POR", 1.28, 1, 1, "W"),
    ("ENG 2-3 ESP",  "ESP", 2.20, 1, 0, "W"),
    ("AUT 3-1 ISR",  "AUT", 1.60, 1, 1, "W"),
    ("KOS 1-0 IRL",  "KOS", 2.50, 0, 1, "W"),
    ("BEL 0-1 FRA",  "FRA", 2.30, 1, 0, "W"),
    ("TUR 1-4 ITA",  "ITA", 1.65, 1, 0, "W"),
    ("NIR 0-0 HUN",  "HUN", 2.60, 0, 0, "D"),
    ("GEO 0-0 UKR",  "UKR", 2.30, 0, 0, "D"),
    ("SWE 3-1 POL",  "SWE", 1.80, 0, 1, "W"),
    ("ROU 2-4 BIH",  "ROU", 2.20, 0, 1, "L"),
    ("ARM 2-3 MNE",  "MNE", 2.20, 1, 0, "W"),
    ("LVA 0-0 CYP",  "CYP", 2.60, 0, 0, "D"),
    ("ESP 4-1 CRO",  "ESP", 1.25, 1, 1, "W"),
    ("CZE 0-2 ENG",  "ENG", 1.45, 1, 0, "W"),
    ("SCO 0-3 SUI",  "SUI", 2.30, 1, 0, "W"),
    ("SVN 2-0 MKD",  "SVN", 1.60, 1, 1, "W"),
    ("DEN 2-4 POR",  "POR", 2.20, 1, 0, "W"),
    ("GER 2-0 SRB",  "GER", 1.30, 1, 1, "W"),
    ("GRE 2-2 NED",  "NED", 1.85, 1, 0, "D"),
    ("WAL 2-1 NOR",  "NOR", 1.55, 1, 0, "L"),
    ("BEL 3-0 TUR",  "BEL", 1.50, 1, 1, "W"),
    ("FRA 1-1 ITA",  "FRA", 1.41, 0, 1, "D"),
    ("BIH 1-1 SWE",  "SWE", 2.05, 0, 0, "D"),
    ("HUN 1-0 GEO",  "HUN", 2.20, 0, 1, "W"),
    ("POL 6-0 ROU",  "POL", 1.55, 1, 1, "W"),
    ("UKR 0-3 NIR",  "UKR", 1.90, 0, 1, "L"),
    ("LVA 1-2 MNE",  "MNE", 1.75, 1, 0, "W"),
]

res = np.array([m[5] for m in M])
dec = np.array([m[2] for m in M])
tier = np.array([m[3] for m in M])
home = np.array([m[4] for m in M])
win = (res == "W").astype(float)


def block(name, mask):
    m = mask.astype(bool)
    n = m.sum()
    if n == 0:
        return
    wr = win[m].mean()
    dr = (res[m] == "D").mean()
    lr = (res[m] == "L").mean()
    imp = (1 / dec[m]).mean()
    roi = np.where(win[m] == 1, dec[m] - 1, -1).mean()
    print(f"{name:34s} {n:3d} {wr:6.1%} {dr:6.1%} {lr:6.1%} {imp:7.1%} {roi:+7.1%}")


print("=" * 86)
print("축구(UEFA NL 1~3R, 36경기) — 페이브 기준 성적")
print("=" * 86)
print(f"{'조건':34s} {'N':>3s} {'승':>6s} {'무':>6s} {'패':>6s} {'암시승률':>7s} {'ML ROI':>7s}")
print("-" * 86)
block("전체 페이브", np.ones(len(M)))
block("홈 페이브", home == 1)
block("원정 페이브", home == 0)
print()
block("티어갭 있음 (실력차 명확)", tier == 1)
block("  티어갭 + 원정", (tier == 1) & (home == 0))
block("  티어갭 + 홈", (tier == 1) & (home == 1))
block("티어갭 없음 (홈빨/서사뿐)", tier == 0)
block("  티어갭X + 홈", (tier == 0) & (home == 1))
print()
block("단배 <=1.50", dec <= 1.50)
block("1.50~1.90", (dec > 1.50) & (dec <= 1.90))
block(">1.90", dec > 1.90)
print()
block("티어갭X & 단배<=1.75 (함정구간)", (tier == 0) & (dec <= 1.75))

print()
print("=" * 86)
print("결론 / 파라미터 보정")
print("=" * 86)
tg = (tier == 1)
ntg = (tier == 0)
print(f"티어갭 있는 페이브 : {win[tg].mean():.1%} 승  (암시 {np.mean(1/dec[tg]):.1%})  "
      f"ROI {np.where(win[tg]==1, dec[tg]-1, -1).mean():+.1%}")
print(f"티어갭 없는 페이브 : {win[ntg].mean():.1%} 승  (암시 {np.mean(1/dec[ntg]):.1%})  "
      f"ROI {np.where(win[ntg]==1, dec[ntg]-1, -1).mean():+.1%}")
print()
print("-> S2(티어갭) 유지·강화 / S3(홈+서사만) 강한 음수 유지")
print("-> 1X2는 무승부로 페이브가 깨지므로, 하키 ML과 같은 가격이라도 가치가 낮다")
print("-> 티어갭 없는 단배 1.75 이하는 구조적 함정 구간")

# derive logit adjustments
def adj(mask):
    m = mask.astype(bool)
    emp = win[m].mean()
    imp = np.mean(1 / dec[m])
    emp = min(max(emp, 0.02), 0.98)
    return np.log(emp / (1 - emp)) - np.log(imp / (1 - imp))


print()
print("경험적 logit 보정값 (표본 적음 -> 1/3만 반영 권장):")
for nm, msk in [("티어갭 있음", tier == 1), ("티어갭 없음", tier == 0),
                ("원정 페이브", home == 0), ("홈 페이브", home == 1)]:
    a = adj(msk)
    print(f"  {nm:14s} raw {a:+.3f}   축소(1/3) {a/3:+.3f}")
