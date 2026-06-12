# KERNEL LAB — findings, plan, and the iteration playbook

*Living document. Updated 2026-06-12. Owner: kernel research loop.*

## Current standings (gen-2, model fills, 5y replay 2021-2026)

| Kernel | Annual | MaxDD | Sharpe | Worst-12mo |
|--------|--------|-------|--------|-----------|
| **champion_smooth_ddtrim_ivrank** | +32.1% | -11.8% | **2.10** | **+4.8%** |
| champion_kold15_ivrank | +31.1% | -10.1% | 1.92 | +2.5% |
| champion_psi_kold15 | +32.6% | -15.6% | 1.92 | -0.1% |
| production (scale_invariant) | +32.1% | -15.7% | 1.87 | -0.3% |

IV-rank z-scaling is the single most valuable knob found: cuts MaxDD by
a third at zero return cost. Gen-3 (real-fill model, running) decides
promotion; if smooth_ddtrim survives honest fills it takes CHAMPION_KEY.

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
- [ ] Gen-3 (real fills + timing) — RUNNING, decides promotion
- [ ] Gen-4 (forensic knobs above) — build after gen-3 lands
- [ ] honest_walkforward on gen-3 winner before CHAMPION_KEY flip
- [ ] Dashboard phase-2 (kernel label/OOS/why + any new live inputs)
