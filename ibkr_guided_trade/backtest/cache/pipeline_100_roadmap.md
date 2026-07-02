# Data Pipeline → 100/100 roadmap (worked by the 15-min improvement cron)

## SCORECARD (5 dimensions × 20 = 100). 100/100 is GATED: it is UNREACHABLE unless the
## FIDELITY dimension's no-leak test PASSES and EIA events are placed at their exact release instant.
CURRENT SCORE: 95/100
  · data-correctness      20/20
  · refresh/monitoring    20/20
  · fill-fidelity         20/20
  · live==backtest parity 17/20
  · FIDELITY (no-leak + minute accuracy)  18/20   <-- NEW gating criterion (raised the bar; see below)

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

## ============ FIDELITY: NO-LEAK + MINUTE ACCURACY (12 → 20) — THE GATE ============
## NEW VISION (operator-authorized 2026-07-01): the loop now BUILDS the minute-reactive, event-exact,
## no-leak backtest in staged sub-steps toward 100/100 — no longer operator-gated. HARD RULE stands: at
## minute T see ONLY data with availability-ts <= T; EIA storage effective at EXACTLY Thu 10:30 ET; monthly
## at its exact release datetime. Each stage: implement → safety + test_no_lookahead green → commit. If a
## stage cannot be safely completed/validated in one fire, split it smaller or PAUSE + report — NEVER ship
## leaky or half-working minute-reactive code. Points credited only when the stage's assertion PASSES.
- [x] DONE Fi1  NO-LOOK-AHEAD ASSERTION TEST (test_no_lookahead.py). (+4)  [earned]
- [x] DONE FiA  release_ts() availability-timestamp map: a pure function giving the exact public moment of
       each series' value — storage=Thu 10:30 ET of release week, monthly EIA=its release date, prices=bar
       ts. Standalone + unit-tested (no engine change yet). The backbone the reactive loop gates on. (+1)
- [x] DONE FiB  Extend test_no_lookahead.py with an EVENT-EXACT assertion: a decision timestamped 10:29 ET
       on a storage-release Thursday must NOT see that day's 10:30 number (uses release_ts). Fails on leak. (+2)
- [~] STAGE FiC  Minute-reactive decision path (engine, param-gated `reactive_events`, default OFF = champion
       byte-identical): on EIA-release days re-evaluate at the 10:30 print using minute spot + newly-released
       storage, ts<=T gated via release_ts. Scoped to event windows (tractable). (+3 total; SPLIT below)
    - [x] DONE FiC-1  Intraday event-moment SPOT source (intraday_spot.reconstruct_spot): PG has NO intraday
           UNG spot (etf_spot_minute=SPY/QQQ/IWM only; ung_options_history.underlying_price is daily-constant).
           Recover it from near-ATM put-call parity S=C_mid-P_mid+K*DF, median across strikes. Empirically
           cross-strike rel_std ~0.1-0.6% (strikes AGREE → real price). test_no_lookahead asserts (1) reliability
           rel_std<2% and (2) causality: 10:30 value is deterministic & != 16:00 (no EOD leak). (+1) [earned]
    - [x] DONE FiC-2  Wire reconstruct_spot into the engine behind param `reactive_events` (default OFF).
           On storage-release Thursdays the day's decision uses the 10:30 event spot (propagates through strike
           select / greeks / NAV / TP). Batch-precomputed before the loop (event_spot_map, one PG pass).
           PROVEN byte-identical: champion default==reactive_events=False bit-for-bit (md5 5dbe629a, 4116
           trades); reactive_events=True differs (4667 trades) = wired. test_no_lookahead asserts the byte-
           identical guard. NO-LEAK: 10:30 spot is public at 10:30 & EARLIER than EOD (strictly less info). (+1)
           CAVEAT: NAV delta (876k->1099k) is NOT validated alpha — decision uses 10:30 spot but fills/marks
           are still EOD (decide-10:30/fill-16:00 inconsistency). FiC-3/FiD must make fills+marks consistent
           and audit before any performance claim. [earned: wiring only]
    - [x] DONE FiC-3a  Consistent reactive fills + fill leak-audit: reactive-mode fills on storage-release
           Thursdays route through execute_audit(exec_window=11, avoid_print=True) → a REAL post-print minute
           quote (decided at the 10:30 print, executed once settled), killing the decide-10:30/fill-16:00 gap.
           test_no_lookahead asserts the fill exec_time is >=11:00 & same-day (never front-runs the print, never
           borrows a later day). Byte-identical off (guard in exec_fill). Full safety green (9+97). (+1)
    - [ ] FiC-3b  PARITY: extend the determinism test so the reactive backtest reproduces the live path's
           same-day 10:30 decision by construction. Unlocks parity 17->20 (+3). (Also: honest reactive
           performance re-measure with consistent fills — quantify how much of the FiC-2 +25% was fill timing.)
- [ ] STAGE FiD  Minute-path fills (intraday_exec) as the default WITHIN reactive mode; model fallback
       off-grid. Backtest fills == live fills. (+2)
- (Parity 17→20 is earned WITH FiC/FiD: the reactive backtest reproduces the live path's same-day decision
  by construction — extend the determinism test to assert it. Credited under the parity dimension.)

## ============ LIVE==BACKTEST PARITY (13 → 20) ============
- [x] DONE P4  Reconcile = accuracy-only: never `continue`/drop an order; show stale/loss warning inline. (+2)
- [x] DONE P3  TP hysteresis: latch an emitted TP for the session + epsilon margin (reduces flicker WITHIN
       a reactive session; complements Fi2/Fi3). (+2)
- (P2 "freeze to close" REJECTED per D1=B — replaced by the FIDELITY project above.)

## ============ FILL FIDELITY (14 → 20) ============
- [x] DONE F1 (D2=A approved) use_real_chain_fills on base regime_wheel_boxx_greeks (default going forward);
       reported headlines drop 20.4→18.4 (intended). Re-run honest_walkforward to confirm. (+2)
- [x] DONE F2  Remove model-fallback open/close asymmetry (buy-side markup on buyback, mirror fill_factor). (+2)
- [x] DONE F3  Route CALL_GAMMA_CLOSE through exec_fill; extend reconcile to SELL opens + roll legs. (+2)

## ============ DATA CORRECTNESS (17 → 20) ============
- [x] DONE D2  EIA monthly release lag: `.shift(21)` under-lags EIA-914 (~2mo). Fix to ~`.shift(42)` (or
       index by release date). Re-validate champion unchanged. (Overlaps Fi1's monthly assertion.) (+1)
- [x] DONE D4  Health check: staleness by LAST-CHANGE not last-row (carried-forward EIA masks a freeze). (+1)
- [x] DONE D3  Remove/populate dead columns (ng_ma200/ng_trend all-NaN, iv_*d proxy, dead COT fetch). (+1)

## ============ REFRESH/MONITORING (18 → 20) ============
- [x] DONE R1  Surface pipeline_health_status.json as a red/yellow/green banner in kernel_dashboard.py. (+2)
