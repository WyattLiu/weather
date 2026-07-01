# Data Pipeline → 100/100 roadmap (worked by the 15-min improvement cron)

CURRENT SCORE: 78/100  (data 21 · refresh 23 · fill 18 · parity 16)

Rules for the cron worker:
- Do ONE `[ ] AUTO` item per fire: implement → validate (safety suite `pytest backtest/test_engine_safety.py
  test_live_kernel_safety.py` must stay green; champion FULL/TEST must not silently change unless the item
  intends it) → commit → mark `[x]` and update CURRENT SCORE → report the delta.
- NEVER apply `[!] DECISION` items — report them to the user and move on.
- If all AUTO items are done, report the final score + the remaining DECISION items, then stop.
- Champion = STRATEGIES['regime_wheel_boxx_greeks_live']. Keep param-gated defaults byte-identical.

## PARITY (16 → 25)  — the operator's flagged "suggestions inconsistent" issue
- [ ] AUTO  P3 TP hysteresis: once a TP is emitted in a session, latch it; gate emission with an epsilon
       margin (cv < entry*thr*(1-eps)) so a few-cent spot/IV move can't flip the order set. (+3)
- [ ] AUTO  P4 Reconcile = accuracy-only: `plan_for_recs`/`_reconcile_economics` must NOT `continue`/drop an
       order; show the stale/loss warning inline but keep it in the set. (+3)
- [!] DECISION P2 Freeze decision inputs to the daily close (order set computed off a once-per-day frozen
       spot/surface snapshot; live spot still drives display/greeks). Biggest parity win but changes live
       decision timing — needs operator sign-off. (+3)

## FILL FIDELITY (18 → 25)
- [ ] AUTO  F2 Remove fallback open/close asymmetry: on the model-fallback path apply a buy-side markup to
       buyback prices (mirror fill_factor for side='buy') so off-grid buybacks aren't optimistically cheap. (+3)
- [ ] AUTO  F3 Route CALL_GAMMA_CLOSE through exec_fill (not raw bs_call), and extend `_reconcile_economics`
       to also reconcile SELL opens + roll legs (currently PUT_TP/CALL_TP only). (+2)
- [!] DECISION F1 Promote honest fills to the REPORTED champion (use_real_chain_fills on base
       regime_wheel_boxx_greeks or repoint CHAMPION_KEY) and re-publish walk-forward headlines — this LOWERS
       reported numbers (20.4→18.4 etc). Needs operator sign-off. (+2)

## DATA CORRECTNESS (21 → 25)
- [ ] AUTO  D2 EIA monthly release lag: `.shift(21)` under-lags EIA-914 (~2mo release). Change production/
       consumption to `.shift(~42)` (or index by release date). Re-validate champion unchanged (monthly
       factors are off in champion). (+2)
- [ ] AUTO  D4 Health check: measure staleness by LAST-CHANGE date, not last-row date (carried-forward EIA
       masks a frozen fetch). Add per-column last-change tracking. (+1)
- [ ] AUTO  D3 Remove dead/misleading columns: ng_ma200/ng_trend (100% NaN), iv_30/60/90d (display-only
       realized proxy mislabeled 'IV'), dead COT fetch — delete or populate + document. (+1)

## REFRESH/MONITORING (23 → 25)
- [ ] AUTO  R1 Surface pipeline_health_status.json as a banner in kernel_dashboard.py (red/yellow/green) so
       the operator SEES a stale feed, not just the log. (+2)
