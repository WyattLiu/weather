# KERNEL LAB — findings, plan, and the iteration playbook

*Living document. Updated 2026-06-12. Owner: kernel research loop.*

## Current standings (COMBINED gen-2/3/4 tournament, 2026-06-12 overnight)

**Model fills (gen-2 basis):**
| Kernel | Annual | MaxDD | Sharpe | Worst-12mo |
|--------|--------|-------|--------|-----------|
| champion_smooth_ddtrim_ivrank | +32.1% | -11.8% | **2.10** | **+4.8%** |
| champion_kold15_ivrank | +31.4% | -10.1% | 1.94 | +2.5% |
| production (scale_invariant) | +32.3% | -15.7% | 1.88 | -0.3% |

**REAL fills (gen-3 basis — the honest numbers):**
| Kernel | Annual | MaxDD | Sharpe | Worst-12mo |
|--------|--------|-------|--------|-----------|
| **g3_kold15_ivrank_rf** | **+27.9%** | **-10.4%** | **1.81** | **+1.6%** |
| g3_psi_rf (champion+RF) | +27.4% | -16.1% | 1.69 | — |
| g3_smooth_ddtrim_rf | +19.0% | -10.1% | 1.67 | -3.5% |
| g3_timing_rf | +15.7% | -7.7% | 1.53 | +1.3% |

### Overnight findings (gen-3/4)
1. **Real fills cost ~5pp/yr** (32.3→27.4 on the champion) — matches the
   fill grid. +27.9%/Sharpe 1.81/MDD -10.4% is the honest production
   expectation for the best kernel (kold15+ivrank under real fills).
2. **Entry-day gating REJECTED (6th filters-law confirmation):**
   Thursday-only put entries cut frequency 5x and returns by 11.7pp
   (27.4→15.7). The microstructure timing edges (~bps) are dwarfed by
   lost premium cycles. Timing stays EXECUTION guidance (place the same
   trades at better hours), never trade-frequency restriction.
   NOTE: test was frequency-confounded (daily base vs weekly-Thursday);
   a fair weekly-vs-weekly-Thursday test remains open for gen-5.
3. **smooth engine is fill-fragile**: smooth_ddtrim_rf 19.0% vs
   kold15_ivrank_rf 27.9% — smooth's edge leans on premium volume that
   real fills tax hardest. kold15_ivrank is the robust champion family.
4. **g4 knobs ran on the timing-crippled base — unreadable except
   relatively**: dd_ivgate +0.8pp (helped), tp_ivrank flat,
   elevator25 flat, rollguards -0.6pp, everything -1.5pp (knob
   interactions negative). RERUN on g3_kold15_ivrank_rf base in gen-5.

### PROMOTION RECOMMENDATION (for the user's morning decision)
Promote **kold15 + iv_rank_z_scale** (strategy
`champion_kold15_ivrank`) as production CHAMPION_KEY:
- Best real-fill profile: +27.9%/1.81/-10.4%/+1.6% floor (as g3_..._rf)
- Dominates current production on MDD (-10.4 vs -16.1 real-fill) and
  floor at comparable return
- One-knob-stack from the live engine → minimal promotion risk
- smooth_ddtrim_ivrank's 2.10 Sharpe is model-fill flattery (drops to
  1.67 under real fills) — do not promote it
OOS gate PASSED: best sealed-test Sharpe 2.16 (vs production 2.07).

### OOS GATE RESULTS (sealed test 2024-01→2026-06, cost model: \$0.65/ct + 5% slippage + assign haircut)
| Strategy | TEST ann | TEST Sharpe | TEST MDD |
|----------|---------|------------|----------|
| **champion_kold15_ivrank** | +31.1% | **2.16** | -9.2% |
| production (scale_invariant) | +33.2% | 2.07 | -8.9% |
| champion_smooth_ddtrim_ivrank | +40.8% | 1.92 | -11.8% |
| g3_kold15_ivrank_rf* | +22.7% | 1.77 | -8.5% |

*g3_rf rows double-count costs (fill grid + walkforward slippage) — read
champion_* rows for the apples-to-apples OOS comparison.

**OOS verdict: champion_kold15_ivrank wins the sealed test on Sharpe
(2.16 vs production 2.07) at comparable MDD. The promotion
recommendation is OOS-VALIDATED.** smooth_ddtrim_ivrank's +40.8% OOS
return is real but with worse Sharpe, deeper MDD, and the gen-3
fill-fragility flag — aggressive-profile alternative only.

### Gen-5 queue (updated 2026-06-13 post-promotion)
1. **DISTRIBUTIONAL DELTA TARGETING (user directive — the rigidity fix):**
   replace point share-targets with a band: target ~ (mu_delta, sigma_delta)
   where mu comes from z x iv_rank (as today) and sigma from SIGNAL
   DISAGREEMENT (z vs iv_rank vs momentum pointing apart = wide band, all
   aligned = tight). Act only when |current - mu| > k*sigma (hysteresis);
   trade toward mu - not onto it. Backtest k in {0.5, 1.0, 1.5}.
2. **CONDITIONAL what-if distributions:** the live what-if (shipped
   2026-06-13) uses the UNCONDITIONAL 35d empirical distribution — it
   flagged today's put mix as EV-negative (-\$16) because it includes
   UNG's structural decay across all regimes. Upgrade: condition the
   scenario windows on the current (z-bucket, iv_rank-bucket) — the
   kernel's whole edge is that conditioning. Compare unconditional vs
   conditional E[PnL] on every rec; large gaps = regime conviction.
3. **Per-day/per-trade forensics on the promoted kernel (user directive):**
   extend trade_forensics with a daily ledger mode — every trading day:
   what was opened/closed/rolled, what the alternative (no-trade,
   assignment-accept) would have done, tag each trade win/loss vs its
   counterfactual. 'What worked / what didn't, each day, each trade.'
4. g4 knobs (dd_ivgate first) rerun on g3_kold15_ivrank_rf base
5. fair timing test: entry_cadence=5 any-day vs entry_cadence=5 Thursday
7. **SHOULDER HEDGE BAKE-OFF (user directives 2026-06-13):** three-way
   comparison over Mar-May/Sep-Nov windows, real fills where possible:
   (a) KOLD shares 15% NAV (current), (b) KOLD shares + covered calls
   on the liquid call side (~12%/70d premium density measured live;
   await historical-liquidity confirmation from KOLD ThetaData backfill
   — after-hours snapshot exaggerated put spreads), (c) UNG long/bear
   puts sized to equivalent hedge delta (cheap when IV-rank low).
   KOLD put-WRITING: judge after historical liquidity study (close
   quotes, not after-hours).
8. **SYNTHETIC LONG / RISK REVERSAL deep analysis (user directive):**
   when IV-rank is low (calls cheap), compare share-accumulation vs
   synthetic long (long call + short put same K — full delta, ~1/5th
   capital) vs risk reversal (short OTM put funds long OTM call).
   Capital efficiency × assignment tails × the covered-calls-only rule
   (synthetic's short put leg is cash-secured anyway). The adapter
   already emits SYNTHETIC_LONG_PARITY candidates — backtest the knob.
9. **Beam analysis now snaps to the \$0.50 real grid** (fixed
   2026-06-13); extend to per-expiry true strike lists later.
10. COST MODEL CORRECTION (user): WS charges NO commission — spread/slippage
   is the only real cost. honest_walkforward's \$0.65/ct is conservative
   padding; keep it as safety margin but report both.


## Forensic findings (6,380 trades, smooth_ddtrim_ivrank)

1. **The put cycle NETS -$45k** (TP +194k, rolls -218k, assigns -21k);
   the CALL cycle nets +$368k. Put-writing is a share-acquisition
   mechanism whose profit lands in the share book — not an income engine.
2. **Roll-downs are the #1 cost line** (-$218k, 388 events) and 28% were
   futile (assigned within 60d anyway — paid to delay).
3. **Worst day was an UP day** (-4.7% NAV on +3.9% spot, 2022-06-27):
   call-side melt-up pain. ELEVATOR_CLOSE averages +$1,504/event vs
   CALL_ROLL_UP -$782 — the good tool fired 85x, the bad one 102x.
4. **Cascade days**: 14 roll-downs in one window (2021-12-09, -4.6% NAV
   on -1% spot) — whole-book rolling into vol-spike spreads.
5. **All >5% drawdowns live in Aug-2021→Feb-2023**; zero since. The
   long episodes (58d/25d/61d) are share-book beta from rich-vol tops.
6. **Put TPs are small** (avg +$167 vs calls +$534) — 50% TP may be
   premature in calm regimes (fast-TP is a high-vol tool per memory).

## Gen-4 candidate knobs (each attacks a numbered finding)

| Knob | Attacks | Spec |
|------|---------|------|
| `roll_accept_cheap_z` | #2 | Skip roll-down, take assignment when z<-0.5 and no falling-knife (kernel wants shares there) |
| `max_rolls_per_chain` | #2 | 1 roll per position then accept — kills the futile 28% |
| `elevator_extrinsic_max: 0.25` | #3 | Widen elevator eligibility to replace roll-ups |
| `roll_stagger_max_per_day` | #4 | ≤3 defensive rolls/day; never the whole book at once |
| `tp_by_iv_rank` | #6 | TP 50% when iv_rank>0.6 (capture before reversion); 70% capture when <0.4 (let calm decay run) |
| `dd_trim_iv_gate` | #5 | Trim faster when iv_rank>0.8 (the -23% fwd zone) |
| spike-day patience | #3 | Defer call roll-ups 1-2d after >3% up-moves (knee-jerks reverse: 5s study r=-.19) |

Design laws (validated this cycle, do not relitigate):
- **One knob per clone** — adjacent entrants differ by exactly one thing.
- **Same fill model for every entrant** in a tournament (real_fill_model
  for gen-3+; mixing fill bases invalidates comparison).
- **Upsize-only for carry tilts**; defensive *gates* must beat their
  head-to-head ablation ([[feedback_filters_cost_more_than_they_save]] —
  5 independent confirmations).
- **Quote both full-sample MDD and worst-12mo** (walk-forward truth).

## THE ITERATION PLAYBOOK (repeat each generation)

```
1. TOURNAMENT     venv/bin/python backtest/replay_engine.py
                  (new candidates in _KEEP_STRATEGIES; background; ~2-3h)
2. FLOOR CHECK    worst-12mo from results/{name}_history.csv
3. FORENSICS      venv/bin/python backtest/trade_forensics.py --strategy <leader>
                  → 6 behavior sections + INTEGRITY/BUG SCREEN (section 7)
4. BUG SCREEN     section 7 must show 0 flags before any result is
                  believed. Known bug families to suspect FIRST whenever
                  a result looks too good:
                    - covered-call stacking (naked shorts) ← bit us 6/11
                    - negative-cash leverage cascades      ← bit us 6/11
                    - liability marking noise (thin quotes)← bit us 6/11
                    - split adjustment (spot vs strikes)   ← bit us 6/11 (x2)
                    - lookahead: publication lags (ONI +1mo, COT +1wk,
                      FPI +2mo, STU vintage), rolling thresholds past-only
                    - fill optimism (BSM vs real bid: 0.67-0.95x at 30-45d)
5. KNOB DESIGN    each forensic finding → one candidate knob; clone the
                  leader + one knob; add an attribution ladder
6. OOS GATE       backtest/honest_walkforward.py --strategies <top 2-3>
                  (sealed test window + cost model) before promotion
7. PROMOTE        flip CHAMPION_KEY in validated_kernel_adapter KERNELS
                  (with fresh OOS metrics + why), restart dashboard,
                  update Executor Brief if the kernel has new live inputs
8. DOCUMENT       update this file: standings, findings, next knobs
```

Cadence: rerun the full loop whenever (a) a new factor validates,
(b) live fills diverge from the fill grid, (c) quarterly at minimum.
Data freshness: fill grid rebuilds from ThetaData snapshots (extend via
research/gex daily collector); IV-rank daily CSV extends the same way.

## Status / queue

- [x] Gen-2 complete — smooth_ddtrim_ivrank leads (2.10 Sharpe, +4.8% floor)
- [x] Gen-3 complete — real fills cost 5pp; kold15_ivrank_rf leads
- [x] Gen-4 complete (on crippled base — rerun queued for gen-5)
- [x] honest_walkforward complete — kold15_ivrank wins OOS (Sharpe 2.16)
- [x] PROMOTED champion_kold15_ivrank (2026-06-13); dashboard phase-2 live (label/OOS/knobs/timing/what-if)
