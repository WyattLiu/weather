# DBA Strategy Research — Consolidated Findings

*As of 2026-06-11. All results reproducible from scripts in this directory.*

## The instrument

DBA (Invesco DB Agriculture): futures basket ~50% grains/livestock
(corn, soy, wheat, cattle, hogs), ~37% softs (sugar, coffee, cocoa).
Low IV (~15% baseline) → theta/$ is 25-50% of UNG's; the leg earns its
keep through **Sharpe and factor tilts, not premium density**.
No splits (unlike UNG — 1:4 reverse 2018-01-05 and 2024-01-24; always
un-adjust yfinance closes before comparing to historical strikes).

## The wheel chassis (wheel_backtest.py)

Cash-secured put-write + covered-call on assignment, weekly entries,
TP at 50% decay. Parameter sweep (25 configs, 2015-2026):

| Config | Ann | Sharpe | MDD | Verdict |
|--------|-----|--------|-----|---------|
| **60d / 2% OTM** | **+17.8%** | **1.47** | -16.7% | production base |
| 45d / 2% | +11.2% | 1.51 | -10.1% | conservative alt |
| 90d / 2% | +28.5% | 0.99 | -48% | rejected — MDD blowout |
| 60-90d / 5% (old kernel rec) | +5-9% | ~1.0 | -11..-34% | too defensive |

## Factor stack (factor_scan.py, fundamentals_scan.py)

Quintile fwd-63d spreads, monthly-sampled. **Significant (p<.05):**

| Factor | Spread | p | n | Sign / mechanism |
|--------|--------|---|---|------------------|
| ONI (Niño 3.4) | -5.7% | .004 | 230 (19yr) | **La Niña bullish** — grain-heavy basket; La Niña wrecks S.American harvests. INVERTED vs naive El Niño-food-crisis thesis (that's only the 12mo softs tail) |
| NG 3m trend | -6.8% | .004 | 54 | macro-cycle echo; n small |
| crude 3m trend | -6.3% | .047 | 54 | same family |
| DXY 3m trend | -6.7% | .020 | 54 | strong dollar bearish ag |
| COT MM flow 13w | -4.2% | .014 | 191 | fast managed-money buying mean-reverts; **level doesn't matter, flow does** (cot_z level: p=.69) |
| USDA stocks-to-use z | -4.2% | .036 | 230 | tight world grain stocks bullish — the ag analog of EIA storage. *Caveat: current-vintage PSD = revision lookahead* |
| FAO FPI 3m momentum | **+3.8%** | .049 | 225 | food-price trends PERSIST (only positive-momentum factor in the stack) |
| Seasonality | — | — | ~20/mo | Dec/Jan/May/Oct strong; Jun/Sep/Nov weak (harvest pressure) |
| DBA price momentum (1-12m) | n.s. | | | nothing |
| IV-rank | n.s. | | | nothing |

## Design law: upsize-only (see feedback_filters_cost_more_than_they_save)

Defensive variants (downsize/skip in bad regime) ALL cut return:
ENSO gate, hh_storm, risk parity, vol-target, dd-responsive — every one
underperformed static. Winning pattern: **floor 1.0×, multiply up** when
factors align (ONI<0 ×1.5, strong month ×1.3, washed-out COT flow ×1.25,
FPI momentum ×1.25, tight stocks ×1.2; cap ~2×).

| DBA variant | Ann | Sharpe | MDD |
|------------|-----|--------|-----|
| baseline 60d/2% | +17.8% | 1.47 | -16.7% |
| oni-upsize only | +19.8% | 1.59 | -12.5% (improves all 3) |
| weather combo + denser | +29.0% | 1.40 | -29% |
| **full-stack combo + denser** | **+32.1%** | 1.36 | -35% |

## Portfolio (composite_empirical.py, allocation_sweep.py)

UNG-kernel × DBA-wheel daily correlation: **+0.06** (≈zero).
Static beats every dynamic scheme tested. With factor-enhanced DBA:

| Portfolio | Ann | Sharpe | MDD | Worst-12mo |
|-----------|-----|--------|-----|-----------|
| UNG kernel only | +32.4% | 1.81 | -14.9% | +0.3% |
| 70/30 × weather-combo DBA | +31.9% | 2.18 | -12.3% | +6.3% |
| 70/30 × full-stack DBA (est) | ~+32% | ~2.2 | — | — |

**Conclusion: the DBA leg no longer costs return. The blend matches
kernel-only return with +0.4 Sharpe and 1/3 less drawdown.**

## Live wiring

- `composite_edge.py` → `dba_wheel_tilt` in composite_state.json (daily
  18:00 cron), consumed by `validated_kernel_adapter.py` SELL_PUT_DBA
  (standing 40% NAV × tilt; consult-only until chain resolver built).
- 48h staleness guard suppresses DBA signals on stale data.
- GEX: wall = CC strike floor only (74% vs 69% hold rate). No pin, no
  vol signal on UNG. DBA GEX unexplored (data on disk).

## Open questions / next research

1. COT flow + FPI momentum tilts not yet in live `dba_wheel_tilt`
   (weather-only now) — needs weekly COT refresh in cron.
2. Vintage USDA data (WASDE release archive) to kill the stu_z lookahead.
3. DBA term-structure/roll-yield factor (needs futures curve history —
   ThetaData has options but not futures curves).
4. Strong-El-Niño 12mo softs tail (the original thesis) as a separate
   LEAP overlay — distinct from the wheel.
5. DBA GEX walls (data already backfilled, never scanned).
