# Data Pipeline → 100/100 roadmap (worked by the 15-min improvement cron)

## SCORECARD (5 dimensions × 20 = 100). 100/100 is GATED: it is UNREACHABLE unless the
## FIDELITY dimension's no-leak test PASSES and EIA events are placed at their exact release instant.
CURRENT SCORE: 77/100
  · data-correctness      18/20
  · refresh/monitoring    18/20
  · fill-fidelity         14/20
  · live==backtest parity 15/20
  · FIDELITY (no-leak + minute accuracy)  12/20   <-- NEW gating criterion (raised the bar; see below)

Rules for the cron worker:
- Do ONE `[ ] AUTO` item per fire: implement → validate (safety suite `pytest backtest/test_engine_safety.py
  test_live_kernel_safety.py` green; NO look-ahead introduced) → commit → mark `[x]` + update CURRENT SCORE →
  report the delta. NEVER apply `[!] DECISION`/`[!] PROJECT` items — report them for the operator.
- Champion = STRATEGIES['regime_wheel_boxx_greeks_live']; keep param-gated defaults byte-identical.

## OPERATOR DECISIONS (2026-07-01)
- D1 = B: live STAYS reactive; make the BACKTEST reactive to match (do NOT freeze to daily close).
- D2 = A: honest numbers everywhere (F1 approved → AUTO).
- MANDATE: parity via a MINUTE-REACTIVE, EVENT-EXACT, NO-LOOK-AHEAD backtest. At minute T see only data with
  availability-timestamp <= T. EIA storage injected EXACTLY at Thu 10:30 ET; monthly EIA at its exact
  release datetime. Minute-level execution fidelity (real bid/ask path). NEVER leak the future.

## ============ FIDELITY: NO-LEAK + MINUTE ACCURACY (8 → 20) — THE GATE ============
- [x] DONE Fi1  NO-LOOK-AHEAD ASSERTION TEST (test_no_lookahead.py, add to safety suite): for each series,
       assert its value is never used before its real release timestamp — storage effective only >= Thu
       10:30 ET of release week (daily proxy: .shift(5) verified correct), monthly only >= its release date.
       Fails the build if any signal front-runs a print. This is the no-leak backbone; earn it first. (+4)
- [!] PROJECT Fi4  (was AUTO; reclassified 2026-07-01) Minute-path execution default — EVAL showed it is
       too slow standalone (>200s/backtest) AND only consistent once decisions are minute-reactive. Folded
       into the Fi2/Fi3 minute-reactive project. Daily backtest correctly stays on real_chain EOD fills. (+2)
- [!] PROJECT Fi2  EVENT-EXACT release placement at MINUTE granularity: storage value becomes visible at the
       exact 10:30 ET print (not day-open), monthly at exact release datetime; timestamp-gate all series. (+3)
- [!] PROJECT Fi3  MINUTE-REACTIVE decision loop on event/exec windows (react to the real post-print minute
       path same-session), preserving ts<=T gating end-to-end. Parity by construction. (+3)

## ============ LIVE==BACKTEST PARITY (13 → 20) ============
- [x] DONE P4  Reconcile = accuracy-only: never `continue`/drop an order; show stale/loss warning inline. (+2)
- [ ] AUTO  P3  TP hysteresis: latch an emitted TP for the session + epsilon margin (reduces flicker WITHIN
       a reactive session; complements Fi2/Fi3). (+2)
- (P2 "freeze to close" REJECTED per D1=B — replaced by the FIDELITY project above.)

## ============ FILL FIDELITY (14 → 20) ============
- [ ] AUTO  F1 (D2=A approved) use_real_chain_fills on base regime_wheel_boxx_greeks (default going forward);
       reported headlines drop 20.4→18.4 (intended). Re-run honest_walkforward to confirm. (+2)
- [ ] AUTO  F2  Remove model-fallback open/close asymmetry (buy-side markup on buyback, mirror fill_factor). (+2)
- [ ] AUTO  F3  Route CALL_GAMMA_CLOSE through exec_fill; extend reconcile to SELL opens + roll legs. (+2)

## ============ DATA CORRECTNESS (17 → 20) ============
- [x] DONE D2  EIA monthly release lag: `.shift(21)` under-lags EIA-914 (~2mo). Fix to ~`.shift(42)` (or
       index by release date). Re-validate champion unchanged. (Overlaps Fi1's monthly assertion.) (+1)
- [ ] AUTO  D4  Health check: staleness by LAST-CHANGE not last-row (carried-forward EIA masks a freeze). (+1)
- [ ] AUTO  D3  Remove/populate dead columns (ng_ma200/ng_trend all-NaN, iv_*d proxy, dead COT fetch). (+1)

## ============ REFRESH/MONITORING (18 → 20) ============
- [ ] AUTO  R1  Surface pipeline_health_status.json as a red/yellow/green banner in kernel_dashboard.py. (+2)
