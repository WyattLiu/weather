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

### GEN-8 OOS GATE (sealed 2024-26, real fills, matched shares) — FINAL

| (matched, real fills) | TEST ann | TEST Sharpe | TEST MDD |
|---|---------|------------|----------|
| g8_baseline_matched | +22.7% | 1.78 | -8.5% |
| **g8_kold_matched** | +22.4% | **2.10** | **-6.6%** |
| g8_kold_light (half) | +22.3% | 1.85 | -8.4% |
| effect (full hedge) | -0.3pp | **+0.32** | **+1.9pp** |

DECISIVE: the edge HELD OUT-OF-SAMPLE. The OOS window (2024-26) EXCLUDES
the big in-sample 2023 edge, yet the matched hedge still delivered +0.32
Sharpe and 1.9pp LESS drawdown for -0.3pp return. Not a 2023 artifact.
Note: kold_light (frac 0.25) does NOT hold OOS (Sharpe 1.85) — the FULL
frac 0.5 is what works.

### PROMOTION BAR — ALL FOUR CRITERIA PASS
1. Matched edge real? YES — confound-free +0.32-0.54 Sharpe at equal shares.
2. Bootstrap significant? YES — 90% CI [+0.19,+1.01] excludes 0.
3. OOS holds? YES — +0.32 Sharpe / +1.9pp MDD in sealed 2024-26.
4. Cost-positive? YES — only -0.3pp return for the protection.

### FINAL RECOMMENDATION (gen-8): PROMOTE the KOLD book hedge
Add kold_book_hedge=True, kold_book_frac=0.5, hedge_sizing_neutral=True
to the production kernel (champion_kold15_ivrank → champion_kold15_ivrank_kbh).
This is the FIRST frontier-improving result that survived full rigor:
removes the gen-7 confound, statistically significant, OOS-validated, and
~free on return (-0.3pp). Real-world champion under real fills is
~22.7%/1.78/-8.5%; with the hedge ~22.4%/2.10/-6.6% — same return, +0.32
Sharpe, ~2pp less drawdown. Strictly better risk-adjusted at no return
cost. Contrast: the band cost ~5pp return for its MDD cut; this costs 0.3pp.
Caveat: benefit is grind-regime-weighted (cheap insurance that pays in
deep-grind years, ~breakeven in calm/spike) — but it held OOS, so deploy
as a STANDING overlay, not tactical. USER DECISION to flip CHAMPION_KEY.

### GEN-9 QUEUE
1. (if promoted) wire kold_book_hedge into validated_kernel_adapter live
   sizing + dashboard Executor Brief (show current KOLD hedge target vs book).
2. tune kold_book_frac between 0.4-0.6 (0.5 works, 0.25 doesn't — find the knee).
3. risk-reversal low-IV-rank overlay (still unbuilt; live-relevant at IV-rank 0).
4. hedge B (KOLD covered calls) to fund the KOLD bleed — could turn the
   -0.3pp cost POSITIVE (the -37.9%/yr decay funds CC income on the hedge).

### GEN-8 CONTROLLED RE-TEST (2026-06-14) — confound removed, rigor applied

DATA INTEGRITY: PASS. Replay KOLD corr -0.984 / beta -1.82 vs UNG
(genuine -2x inverse); decays -37.9%/yr, captured. (gen-7 nan was an
analysis-script bug, not the sim.)

CONTROL VALID: g8_kold_matched vs g8_baseline_matched hold equal shares
(8,179 vs 8,142, +0.5%) via hedge_sizing_neutral → confound removed.

CONFOUND-FREE VERDICT (equal shares):
| | Return | Sharpe | MaxDD |
|---|--------|--------|-------|
| matched_baseline | +29.1% | 1.87 | -10.5% |
| matched_hedge | +28.8% | 2.41 | -9.9% |
| hedge effect | -0.3pp | **+0.54** | +0.6pp |

The TRUE hedge effect (vs the gen-7 confounded +0.25): +0.54 Sharpe for
-0.3pp return. Effect is daily-VOL smoothing (the -2x inverse dampens
swings) more than tail-MDD (only -0.6pp). Confound had MASKED half the
real Sharpe effect (more shares inflated the gen-7 denominator).

RIGOR CHECKS:
(a) REGIME-STRATIFIED — the critical caveat: the +0.54 is ~ALL from 2023
    (+1.98 edge, the deep-grind year). Other years: 2021 -0.57, 2022
    +0.09, 2024 +0.15, 2025 +0.35, 2026 -0.19. NOT all-weather — pays
    big in deep-grind regimes, mildly negative in spike/calm.
(b) BOOTSTRAP — 90% CI on Sharpe diff [+0.19, +1.01], EXCLUDES 0 →
    statistically significant in-sample (but driven by 2023 blocks).
(c) COST — $16.6k/yr KOLD bleed paid, offset to -0.3pp net return:
    cheap regime-insurance, ~breakeven on return, Sharpe-positive.
(d) Forensics: 0 integrity flags.

HONEST CONCLUSION (pending OOS /tmp/walkforward_g8.log): the KOLD book
hedge at matched shares is a REAL, significant, ~free Sharpe improver —
but it is REGIME INSURANCE concentrated in deep-grind years, not an
all-weather edge. Promote ONLY if OOS (2024-26, the non-2023 window)
shows the edge holds forward; if OOS edge ~0, it is documented as
"cheap grind-regime insurance, deploy tactically when a grind regime is
diagnosed" rather than a standing kernel change. Far better than the
band either way (-0.3pp vs -5pp return cost).

### GEN-7 RIGOROUS REVIEW (2026-06-14) — DOWNGRADE: "win" NOT proven
A professional regime+trade+confound review demolishes the easy verdict:
1. REGIME-CONCENTRATED: per-year Sharpe edge vs baseline 2021 -0.02,
   2022 +0.04, 2023 +0.48, 2024 +0.09, 2025 +0.18, 2026 +0.20. The
   aggregate Sharpe (2.06) is dominated by ONE year (2023 deep-grind).
   Directionally consistent + regime-coherent, but not a uniform win.
2. SHARE-COUNT CONFOUND (fatal to the easy read): KOLD-hedge holds 22%
   MORE shares (10,992 vs 9,037) but earns LESS per share (2.12 vs 2.67
   bp/10k-sh-yr). The aggregate Sharpe lift is partly just more
   exposure, not hedge efficiency. Must control share count to attribute.
3. MECHANISM UNVERIFIED: corr(KOLD,UNG) returned nan; KOLD price path in
   the sim could not be validated to move inverse to UNG. CANNOT prove
   the hedge hedged. Possible KOLD-pricing/alignment bug in replay.
VERDICT: promising in the grind regime ONLY; confounded + unverified.
DO NOT promote. The funded-collar/floor failures stand (grinds != crashes).

### GEN-8 QUEUE (controlled re-test — quant discipline)
1. VERIFY KOLD pricing in replay_engine (spot_k path): confirm the sim
   marks KOLD at real inverse-correlated prices, not static/NaN. BLOCKING
   — nothing about KOLD is believable until this passes.
2. CONTROLLED hedge test: KOLD-hedge vs baseline at MATCHED share targets
   (force identical share book; only difference = the hedge overlay) to
   isolate hedge effect from exposure. This is the ONLY fair test.
3. Regime-stratified OOS: report per-regime (spike/grind/calm) Sharpe+MDD
   separately, not blended — demand the edge holds in the grind regime
   out-of-sample, accept neutrality elsewhere.
4. Block-bootstrap the Sharpe difference — is 2.06 vs 1.81 inside noise?
5. Only after 1-4: revisit gen-7 hedge promotion. Until then champion
   kold15_ivrank stands; band documented as risk-off alt.

### GEN-7 RESULTS (book hedges, real fills) — frontier-break test

Baseline g7_baseline_rf: +27.9% / MDD -10.4% / Sharpe 1.81 / floor +1.6% / 9,037 sh

| Hedge | Return | MDD | Sharpe | Floor | Shares | VERDICT |
|-------|--------|-----|--------|-------|--------|---------|
| **g7_kold_bookhedge** | +27.3% | **-9.6%** | **2.06** | **+5.4%** | 10,992 | **WINS** |
| g7_combo_collar_kold | (see OOS) | — | — | +4.1% | 9,682 | partial |
| g7_funded_collar | +28.8% | -10.9% | 1.78 | +1.9% | 9,755 | NO-HELP (MDD worse) |
| g7_collar_aggr | +29.1% | -10.8% | 1.78 | +3.2% | 9,912 | NO-HELP (MDD worse) |
| g7_scaled_floor | +27.3% | -10.6% | 1.77 | +0.3% | 8,846 | NO-HELP |
| g7_scaled_floor_hi | +26.4% | -11.2% | 1.70 | — | — | NO-HELP |

**THE LESSON (overturns my own headline prediction): the funded collar
FAILED.** OTM put-spreads insure against CRASHES, but the drawdowns are
moderate GRINDS (the 2023-26 chop on a big book), so the protection
never activates and just drags theta — MDD actually got slightly WORSE.
The KOLD book hedge WON because a continuously-offsetting 2x-inverse
position cancels continuous grind drawdowns — right tool for the actual
risk shape. Hedge FORM matters more than hedge presence.

**g7_kold_bookhedge vs baseline: -0.6pp return for -0.8pp MDD, +0.25
Sharpe, +3.8pp floor — AND holds MORE shares (10,992 vs 9,037)** (fewer
drawdowns → dd_trim fires less → book stays fuller, virtuous cycle).
This is a far better frontier point than the band (band gave up ~5pp
return for its MDD cut; KOLD hedge gives up 0.6pp). It does NOT reach
the band's -8% MDD, but at a tiny fraction of the cost. Forensics 0
flags. OOS gate running.

### GEN-7 RUNNING (2026-06-13): BOOK HEDGES — keep shares, hedge the book

DIAGNOSIS (corrected, data-backed): the drawdowns are NOT over-exposure
at tops — they are 84%-UNCOVERED share-book beta in 2023-26 (the kernel
holds avg 12,594 shares but covers only 16% with calls; 84% naked-long
takes full UNG downside). The gen-5/6 band only fixed this by holding a
smaller book (crude → costs return). Gen-7 hedges the book and KEEPS the
shares. Goal: the band's -4.8% MDD WITHOUT its ~5pp return cost.

7 candidates, one knob per clone on promoted kernel + real fills, vs
g7_baseline_rf (same, no hedge) — clean attribution:
- g7_funded_collar / g7_collar_aggr: 10%-OTM put-SPREAD on the uncovered
  book, funded by recent CC premium (~0 net cost). THE HEADLINE — only
  angle that can break the frontier (protection without cutting shares).
- g7_scaled_floor / _hi: protective puts sized to the UNCOVERED book
  (vs token tail_hedge_floor). Cheapest at low IV-rank (now).
- g7_kold_bookhedge: KOLD shares scaled to offset the uncovered UNG book
  year-round (2x inverse).
- g7_combo_collar_kold: collar + KOLD stack.

Tracked one-by-one: each must reduce MDD vs g7_baseline_rf WITHOUT
giving up the return (that is the frontier-break test). OOS gate
mandatory. Smoke: all three hedge types fire correctly.

### GEN-6 OOS GATE (sealed test 2024-2026, real fills + costs) — FINAL

| Kernel | TEST ann | TEST Sharpe | TEST MDD |
|--------|---------|------------|----------|
| **g6_cb_a20_b10** | +17.9% | 2.43 | -4.8% |
| g6_cb_a20_b30 | +17.1% | 2.49 | -4.4% |
| g6_cb_tightfloor | +16.7% | 2.43 | -4.5% |
| g5_band_k15 (plain band) | +16.6% | 2.51 | -4.4% |
| champion_kold15_ivrank (model fills) | +31.1% | 2.16 | -9.2% |
| [champion under REAL fills = g5_promo_rf] | +22.7% | 1.77 | -8.5% |

**No overfitting:** all conviction cells cluster (Sharpe 2.43-2.49, MDD
-4.4 to -4.8%); in-sample leaders stayed OOS leaders. The conviction
band's return-recovery shows OOS too — g6_cb_a20_b10 recovers +1.3pp
OOS return (17.9 vs plain band 16.6) at near-identical Sharpe/MDD.
Confirmed: the design works, modestly.

### FINAL PROMOTION DECISION (apples-to-apples, all real fills)
| | Return | Sharpe | MaxDD |
|---|--------|--------|-------|
| Current production (kold15_ivrank, real-world) | ~22.7% | 1.77 | -8.5% |
| Best band (g6_cb_a20_b10) | ~17.9% | 2.43 | -4.8% |

The band trades **~5pp OOS return for +0.66 Sharpe and 44% less
drawdown.** This is a pure risk-appetite choice:
- **USER HAS REPEATEDLY STATED A RETURN PREFERENCE** → KEEP current
  champion_kold15_ivrank. The band's edge is risk-adjusted, not return.
- If capital-preservation / smoothness becomes the priority → promote
  g6_cb_a20_b10 (best return among the band family, top-tier Sharpe/MDD).
- Conviction refinement over plain band is real but marginal (+1.3pp).

RECOMMENDATION: hold current champion; keep g6_cb_a20_b10 documented as
the risk-off alternative. The band research is COMPLETE — diminishing
returns on further band tuning.

### GEN-7 QUEUE (move off band tuning — exhausted)
1. Hedge structure B (KOLD shares + covered calls) wired for the Sept
   shoulder window — ~+3.6%/yr honest rent, the clearest unbanked win.
2. Risk-reversal overlay at low IV-rank (better tail, 1/5 capital;
   live-relevant NOW at IV-rank 0.00).
3. Separate the band's hold vs half-step effects (mechanism clarity).
4. Conditional what-if distributions (condition scenarios on z/iv_rank
   bucket — today's live what-if is unconditional).

### GEN-6 RESULTS (conviction-scaled band) — 2026-06-13

| Kernel | Annual | MaxDD | Sharpe | Floor | avg shares |
|--------|--------|-------|--------|-------|-----------|
| g5_promo_rf (band off) | +27.9% | -10.4% | 1.81 | +1.6% | 9,037 |
| g5_band_k10 (plain band) | +27.5% | -8.0% | 2.22 | +3.9% | 5,505 |
| **g6_cb_a20_b10** | +27.7% | -8.0% | 2.18 | +3.5% | 5,978 |
| g6_cb_a20_b30 | +27.0% | -7.8% | 2.23 | +4.5% | 5,144 |
| g6_cb_tightfloor | +27.0% | -7.5% | 2.19 | +5.1% | 5,225 |
| g6_cb_a50_b30 | +25.4% | -7.2% | 2.12 | — | — |

**Findings:**
1. **Conviction scaling works as designed but is a REFINEMENT, not a free
   lunch.** g6_cb_a20_b10 holds 5,978 shares vs the plain band's 5,505
   (recovers exposure at extremes) → +0.2pp return at equal MDD/Sharpe.
   The (a,b) cells trace the SAME return/MDD frontier as the plain band's
   k — narrower (a20_b10) = more return+exposure, wider (a50_b30) = lower
   MDD. No cell beats the frontier; conviction is a smoother dial on it.
2. **IMPORTANT CORRECTION to the gen-5 brief:** the "-14pp return" cost
   of the band was MISLEADING — it compared band (real fills) to
   champion_kold15_ivrank (MODEL fills), conflating the ~5pp real-fill
   haircut with the band effect. The HONEST band cost vs g5_promo_rf
   (both real fills) is only **-0.4pp full-sample / ~-6pp OOS** for
   HALVING the drawdown. The band is a much better deal than first framed.
3. Floors improve monotonically with band width (+1.6% no-band →
   +5.1% tightfloor). All integrity screens 0 flags.

### GEN-6 RUNNING (2026-06-13): conviction-scaled band
Headline: band width = base x [floor + a*(1-|consensus|) + b*disagreement]
— NARROW at extremes (recover the 14pp return lost to over-damping the
cheap-z accumulation that drives 41% of rallies), WIDE at neutral (keep
the halved-MDD benefit, no churn for no edge). Supersedes the uniform
shallow-band idea. 10 candidates: 3x3 (a,b) sweep + tight-floor variant,
vs g5_band_k10/k15/promo baselines. TWO free params → OOS gate mandatory
(coarse sweep, demand the edge holds out-of-sample not just in-sample).
Mechanism target: cut the '39% fewer shares on average' down to fewer-
only-at-neutral, restoring extreme-z conviction sizing.

### GEN-5 OOS GATE (sealed test 2024-01→2026-06, real fills + cost model)

| Kernel | TEST ann | TEST Sharpe | TEST MDD |
|--------|---------|------------|----------|
| **g5_band_k15** | +16.6% | **2.51** | **-4.4%** |
| g5_band_k10 | +17.8% | 2.50 | -4.5% |
| g5_band_k05 | +18.1% | 2.35 | -5.0% |
| g5_promo_rf | +22.7% | 1.77 | -8.5% |
| champion_kold15_ivrank (PRODUCTION) | +31.1% | 2.16 | -9.2% |

**OOS-VALIDATED, with an honest trade-off the user must weigh:**
The delta band is unambiguously better RISK-ADJUSTED out-of-sample —
Sharpe 2.51 vs production 2.16, and MaxDD -4.4% vs -9.2% (HALF the
drawdown), cost drag only 4.9% of NAV. BUT it gives up ~14pp of OOS
raw return (16.6% vs 31.1%). The band trades return for a dramatically
smoother, shallower-drawdown ride.

DECISION FRAMING (user has stated a return preference):
- Want max RETURN → keep current champion_kold15_ivrank (+31% OOS).
- Want best RISK-ADJUSTED / capital preservation → promote g5_band_k15
  or k10 (half the drawdown, +0.35 Sharpe, ~17% return).
- COMPROMISE for gen-6: a SHALLOWER band (k=0.3, or sigma floor lower)
  to keep more return while still damping the worst churn — the current
  bands may be over-damping (return cost > needed). The 5-hold mechanism
  finding suggests the half-step is doing the work; a gentler half-step
  could recover return. THIS IS THE GEN-6 HEADLINE EXPERIMENT.

Gen-6 queue: (1) shallow-band sweep k in {0.2,0.3,0.4} + half-step
fraction knob to recover return; (2) separate band-hold vs half-step
effects; (3) hedge structure B (KOLD+CC) for Sept shoulder; (4) risk-
reversal low-IV-rank overlay (live-relevant now, IV-rank 0.00).

### GEN-5 RESULTS (2026-06-13 overnight, real fills)

| Kernel | Annual | MaxDD | Sharpe | Worst-12mo |
|--------|--------|-------|--------|-----------|
| g5_band_k15 | +26.6% | **-7.5%** | **2.23** | **+4.9%** |
| **g5_band_k10** | +27.5% | -8.0% | 2.22 | +3.9% |
| g5_band_k05 | +27.6% | -8.6% | 2.17 | +2.8% |
| g5_promo_rf (PROMOTED, baseline) | +27.9% | -10.4% | 1.81 | +1.6% |
| g5_dd_ivgate | +27.5% | -10.2% | 1.84 | +1.9% |
| g5_tp_ivrank | +27.6% | -10.7% | 1.80 | — |
| g5_rollguards | +16.2% | -21.8% | 1.37 | -15.7% |
| g5_timing_weekly | +20.5% | -9.9% | 1.37 | +2.8% |
| g5_timing_thu | +14.0% | -11.8% | 1.10 | -2.9% |

**Three key questions, answered:**
(a) **DELTA BAND WINS decisively.** All three k beat the promoted
   baseline on BOTH Sharpe (1.81→2.17-2.23) AND MaxDD (-10.4→-7.5/-8.0)
   at ~equal return (-0.3 to -1.3pp). k=1.0 is the balanced pick (2.22
   Sharpe, -8.0 MDD, +27.5%, +3.9% floor); k=1.5 is the risk-min
   (2.23/-7.5/+4.9% floor, -1.3pp return). The band cuts drawdown by a
   QUARTER. Mechanism caveat: only 5 logged BAND_HOLDs — most of the
   gain is the "trade TOWARD mu not onto it" half-step damping, not the
   hold; gen-6 should separate these two effects.
(b) **g4 knobs: still net-neutral-to-negative even on uncrippled base.**
   dd_ivgate +0.03 Sharpe (marginal), tp_ivrank flat, rollguards STILL
   -11.7pp return / -0.44 Sharpe. The forensic hypothesis that rolls
   were "wasteful" is REFUTED: removing/capping rolls (taking assignment
   instead) costs MORE than rolling — assignment is worse than the roll
   in this engine. Roll-downs are a cost but the necessary kind. DROP
   the rollguard knobs permanently.
(c) **Fair timing test (equal weekly cadence): Thursday STILL LOSES**
   (14.0% vs 20.5% weekly-any-day). The gen-3 rejection HOLDS even
   frequency-controlled — Thursday-entry restriction is genuinely
   negative, not a confound. AND weekly cadence itself (<daily) costs
   ~7pp. Timing stays EXECUTION-ONLY guidance, never a kernel gate. 7th
   filters-law confirmation.

### GEN-6 RECOMMENDATION
**Promote g5_band_k10 (or k15 for risk-min) as the next champion** —
pending OOS gate (running /tmp/walkforward_g5.log). The delta band is
the first knob since IV-rank to improve Sharpe AND drawdown
simultaneously, and it directly implements the user's distributional
critique. Drop rollguards/timing knobs permanently. dd_ivgate optional
(marginal). Gen-6 work: (1) separate band-hold vs half-step effects;
(2) widen the band's sigma model (more signals → richer disagreement);
(3) wire hedge-structure B (KOLD+CC, +3.6%/yr rent) for September
shoulder; (4) risk-reversal overlay at low IV-rank (better tail, 1/5
capital — live-relevant now).

### Gen-5 IN PROGRESS (2026-06-13 overnight)
Tournament running: 10 g5_* candidates on the PROMOTED kernel base, real
fills. Headline: DISTRIBUTIONAL DELTA BAND implemented (delta_band_sizing
+ delta_band_k): mu = z x iv_rank target; sigma = signal-disagreement
(z/iv_rank/momentum votes std); hold inside mu +/- k*sigma, trade TOWARD
mu not onto it. Candidates k in {0.5,1.0,1.5}, g4 knobs re-tested on
uncrippled base, fair weekly-vs-Thursday timing test.

Research done tonight:
- HEDGE BAKE-OFF: A KOLD shares +3.5% shoulder contribution (bleeds);
  B KOLD+CC ~+7%/yr standing rent on the hedge sleeve (honest: 50%
  capture x 5 cycles, NOT the naive 100% sum); C UNG bear puts 4.3%/cycle
  theta, event-convex, cheapest at low IV-rank. VERDICT: promote B as the
  standing shoulder hedge; C as opportunistic low-IV-rank overlay.
- SYNTHETIC/RR: per-$ return, synthetic/RR ~= shares (no free lunch on
  return), BUT risk reversal has a BETTER tail at low IV-rank (p5 -17.4%
  vs shares -23%) at ~1/5 capital. Edge = capital multiplier + defined
  floor, not per-$ return. Relevant NOW (IV-rank 0.00).
- KOLD LIQUIDITY (real close quotes, 1959d): put median spread 14.8%
  (NOT the 36-49% after-hours snapshot), tightest in 2022-23 NG crisis
  (4-6%). Regime-dependent — opportunistic only, never standing.

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
- [x] Gen-4 complete
- [x] Gen-5 complete — delta band OOS-validated (half MDD, ~5pp real-fill return cost)
- [x] Gen-6 complete — conviction band = frontier refinement (+1.3pp OOS recovery); band research EXHAUSTED. REC: keep current champion (return pref); band documented as risk-off alt. Gen-7 = hedge B + risk-reversal.
- [x] honest_walkforward complete — kold15_ivrank wins OOS (Sharpe 2.16)
- [x] PROMOTED champion_kold15_ivrank (2026-06-13); dashboard phase-2 live (label/OOS/knobs/timing/what-if)
