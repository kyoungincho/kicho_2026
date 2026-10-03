# Sports Prediction Parameters v2 (optimized)
Updated: 2026-10-03 KST — after backtesting v1's qualitative weights against the
past week's games and refitting them numerically.
Method: situation → review → accumulate → parameterize → next prediction
NOT: odds-first, chalk-first, or dog-first

## What changed from v1 → v2
v1 scored features on an invented 0–3 integer scale, and a "주력" needed ≥ +4.
That scale had no units, so it could not be checked and could not be priced.
v2 replaces it with **logit weights on top of the no-vig market price**, so
every feature's size is measurable and every pick converts to an EV number.

Backtest: NHL 2026-27 **regular season only** (Sep 29 – Oct 2, 21 games) +
UEFA NL matchday 1–3 (36 matches). Code in repo: `model_v4.py`,
`diagnose_v4.py`, `soccer_model_v4.py`, `forward_ev.py`.

## Core loop (unchanged)
1. Situation (lineup, goalie, rest, motivation, H2H script, injuries)
2. Review (what hit/missed and WHY)
3. Accumulate (tag outcomes by feature)
4. Parameterize (weights / hard rules)
5. Predict (features must fire before price)

Price/odds = last check for sizing, never first signal.

---

# PART 1 — NHL (regular season only)

## Model form
```
logit(p_fav) = logit(p_market_novig) + b0 + b_away*(fav is away) + Σ w_i * x_i
p_final      = clip(p, p_market ± 0.09)        # rail
```

## Global terms — fixed from SEASON base rates, not from the 21-game week
| Term | Value | Source |
|------|-------|--------|
| b0 (favorite tax) | **−0.103** | 2025-26 regular season: favorites won 57.3% vs ~59.8% implied |
| b_away (favorite on road) | **−0.050** | home underdogs are the structurally underpriced side |

These two alone improve LOO logloss by +0.0145 over the market with zero
situational input. Effectively: a −170 favorite (63% implied) is really 60.5%,
and 59.3% if it is on the road.

## Situation weights (logit, prior-blended; N0 = 140, ridge λ = 5)
| ID | Feature | w | effect on a 60% team |
|----|---------|------|------|
| N1a | favorite key skater (top-6 F / top-4 D) OUT, per player | **−0.106** | 60% → 57.4% |
| N1b | underdog key skater OUT, per player | **+0.097** | 60% → 62.3% |
| N2a | favorite goalie downgrade (backup / collapsing tandem) | **−0.211** | 60% → 54.8% |
| N2b | underdog goalie downgrade | **+0.191** | 60% → 64.5% |
| N3a | favorite on 2nd of back-to-back | **−0.096** | 60% → 57.7% |
| N3b | underdog on 2nd of back-to-back | **+0.096** | 60% → 62.3% |
| N4 | 노나먹기 (favorite lost the previous meeting) | **+0.076** | 60% → 61.8% |
| N5 | underdog has clear offseason roster upgrade | **−0.076** | 60% → 58.2% |

Goalie downgrade is the single strongest feature, ~2x a skater. That matches
the week's raw splits: favorite goalie downgrade → favorites went 20%.

## Validation (leave-one-out, 21 games)
| Model | logloss | Brier | accuracy |
|-------|---------|-------|----------|
| market only | 0.7213 | 0.2636 | 38.1% |
| + global terms, no features | 0.7068 | 0.2565 | 42.9% |
| **v4 full** | **0.6846** | **0.2461** | **57.1%** |

Selection check: the 10 games v4 bet returned +65.7% ROI (7/10); the 11 games
it **filtered out** returned −1.0% if bet blind. Blind-dog-everything returned
+30.7%. So v4 beats "fade all favorites" by +35pp — it is not just a dog bias.

Permutation test (1000 shuffles of the feature rows): real feature contribution
+0.0222 logloss vs random mean +0.0022, **p = 0.080**. Weak but real. Trust the
feature *signs*; do not trust their *magnitudes* — hence N0 = 140, which keeps
the weights near their priors.

## Hard rules (survived the backtest)
- **Rail ±9pp.** Never move more than 9 points off the market price.
- **EV floor +4%** to bet at all. Below that → PASS, value = 0.
- **Half-Kelly, 4% of bankroll max per bet.**
- **Daily exposure cap 12%** of bankroll across the whole board.
- Price ≤ −170 is still never a 주력 on its own; it now simply fails the EV floor.
- Blind dog / blind +1.5 remains forbidden: it scores +30.7% here only because
  this specific week was dog-heavy.

---

# PART 2 — Soccer (UEFA NL / internationals)

## The one parameter that matters: tier gap
| Bucket | N | Win | Draw | Loss | implied | ML ROI |
|--------|---|-----|------|------|---------|--------|
| tier gap present | 20 | 90.0% | 5.0% | 5.0% | 59.6% | **+57.9%** |
| tier gap absent | 16 | 25.0% | 43.8% | 31.2% | 48.0% | **−47.5%** |

1X2 favorites die by draw, not by loss. Without a real quality gap, a short
favorite price is a trap.

## Weights (logit on no-vig price; N0 = 60, λ = 3, rail ±20%)
| ID | Feature | w |
|----|---------|------|
| S2 | clear tier gap | **+0.297** |
| S3 | no tier gap (home advantage / narrative only) | **−0.284** |
| S7 | favorite on the road | **+0.069** (road-favorite discount is overdone) |

**Deliberately shrunk.** A free fit wanted w_tier = +0.95 and claimed 80.6%
accuracy, but I assigned the tier-gap labels *after* seeing results, and LOO
cannot detect that leakage. Temporal holdout (train on first 24, predict last
12) gave the honest number: **+0.035 logloss over market, 66.7% accuracy.**

Soccer EV floor **+10%** (higher than hockey — 1X2 variance is larger).

## Draw: NOT an edge
The 43.8% draw rate in no-gap matches is really 4 non-home-favorite matches that
all drew. No-gap *home* favorites drew only 25.0% vs 29.4% implied. Shrunk draw
estimate = 32.0%, so a draw is only playable at 3.0+ and in small size.
**In a no-tier-gap match the primary action is PASS, not the draw.**

---

# PART 3 — MLB

**No validated model.** No MLB backtest was run (hockey regular season +
soccer only), so MLB produces no EV number and every MLB pick is PASS until
a backtest exists. Regular-season parameters would not transfer to the
postseason anyway: full-bullpen usage and shortened rotations change the
generating process.

---

# PART 4 — Betting value, numerically

```
EV    = p_model * (dec - 1) - (1 - p_model)
f     = min(4%, half-Kelly)                        # 0 if EV < floor
VALUE = f * EV                                     # in bp of bankroll
```
VALUE is the ranking number, not EV. A huge EV at a price Kelly only lets you
back with 0.4% is worth less than a moderate EV you can size properly.

Reference: a 13-game NHL slate + 4 NL matches produced **5 bets worth 300bp
raw / 180bp after the 12% daily cap**. That is the honest expected value of one
good day: about **+1.8% of bankroll**, not a jackpot.

---

## Accumulated case log (features → outcome)

### Hit cases
| Date | Pick | Key features that fired | Outcome |
|------|------|-------------------------|---------|
| 10/1 | GER ML | home + opponent 0pts + structural gap | W 2-0 |
| 10/1 | POR ML | quality gap away | W |
| 10/1 | UTA ML | Bedard IR + Mangiapane OUT + CHI opener loss | W 6-0 |
| 10/1 | EDM ML | 노나먹기 (VAN OT win prior) + talent gap | W 9-7 |
| 10/2 | BEL ML | TUR 0-2 group + home | W 3-0 |
| 10/2 | MNE ML | League C mismatch + form | W 2-1 |
| 10/2 | NYR ML | DET Larkin OUT + Edvinsson OUT (roster hole > home fav) | W 2-0 |
| 10/2 | WSH side | Tuch/Kyrou/Jenner upgrade + CAR Jarvis IR + paper power | W 5-2 |
| 10/2 | STL side | Duchene LTIR + close H2H script + RLM vs public chalk | W 4-0 |
| 10/2 | ANA +1.5 | fade -190 after 1 game sample; playoff rival knowledge | W 4-3 |

### Miss cases
| Date | Pick | Failed assumption | Actual drivers |
|------|------|-------------------|----------------|
| 10/1 | NOR ML | "WAL can't score / clear mismatch" | WAL scored, form narrative broke |
| 10/2 | FRA ML | home chalk + Zidane clean sheets = win | ITA equalized; 1-0 scripts draw risk |
| (hypo) | CAR ML | Cup + home + already played | roster power/injuries ignored |
| (hypo) | VGK -190 | 1-game sample + big price = lock | opener variance |

### User corrections (hard process rules)
1. Don't dismiss motivation blindly.
2. Don't chalk-only.
3. 노나먹기 = rematch series split (fade last H2H winner), not OT/draw.
4. NHL: roster/power reality can beat home-favorite narrative (WSH > CAR read).
5. Odds-alone NHL analysis → accuracy collapses.
6. Dog-first is ALSO wrong — situation first, then parameterize.
7. Qualitative 0–3 scores are unfalsifiable → must be logit weights with EV.
8. Hockey: regular-season data only.

---

## Graded: KST 10/4 pre-optimization board vs post-optimization value
| v1 pick | v1 score | v2 value | verdict |
|---------|----------|----------|---------|
| SEA (vs EDM) | +8 | 81bp | **survives**, EV +20.2% — best bet on the board |
| TB ML | +5 | 0bp | dead, EV +3.1% < floor |
| ESP ML | +5 | 0bp | dead, EV +0.7% — tier gap already fully priced at 1.25 |
| SUI ML | +4 | 0bp | dead, EV +6.5% < 10% floor, and tier gap is debatable |
| MIL G1 | +4 | 0bp | withdrawn, no MLB model |

v1 PASSes that v2 says were **wrong**:
| game | side | EV | value |
|------|------|------|------|
| DAL@NSH | NSH | +15.4% | 62bp |
| CAR@PHI | PHI | +14.9% | 60bp |
| CRO-ENG | ENG | +13.5% | 54bp |

Lesson: v1 passed on these because the price looked unremarkable. v1 had no way
to compare a feature stack against a price. v2 does.

## Open questions for the next review
- Does the b0 favorite tax hold outside opening week, or was 57.3% itself a
  low-favorite season? Re-measure monthly.
- N4 (노나먹기): does it survive a goalie swap between the two meetings?
- Soccer tier gap: must be labeled BEFORE kickoff from rankings/market, never
  after, or the weight inflates again.
- MLB: needs a real backtest before any pick is allowed.
- Is the ±9pp rail ever the binding constraint on a winner? Log when it clips.

## Versioning
- v1: initialized 2026-10-03 from Sep25–Oct2 reviews + user process corrections
- v2: 2026-10-03, backtested + refit. Integer scores replaced by logit weights,
  global favorite tax added, soccer weights shrunk for hindsight bias, MLB
  disabled pending a backtest, value expressed in bp of bankroll.
