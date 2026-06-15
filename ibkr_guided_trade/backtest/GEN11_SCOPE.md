# GEN-11 SCOPE — return from PAYOFF GEOMETRY, not exposure

*Drafted 2026-06-15. Follows the gen-10 verdict.*

## Why this generation exists

Gen-9 (sizing) and gen-10 (bigger hedged book) both bought return by raising
**share exposure** — and both gave it all back out-of-sample (Sharpe collapsed
below 2.0, DD went super-linear). The closing lesson: *exposure-bought return
borrows Sharpe from the future.*

Gen-11 attacks a different lever — the **option leg geometry** (strike depth,
ratios). Changing the payoff shape can add return at the SAME net share delta:
basis improvement, premium harvest on shares already held, and convexity in the
vol-expansion regime the grinder structurally misses. This is the "return the
hedge doesn't have to pay for" that gen-10 lacked.

## HARD CONSTRAINT — read before scoping any "ratio"

**Covered calls only. No naked short calls. No short shares. (standing account rule.)**

A textbook **call ratio spread = short 2 / long 1** → the extra short call is
NAKED → **FORBIDDEN.** It cannot be scoped as-is. `audit.py`'s
`covered_calls_only` check is FATAL and will auto-reject any clone that produces
short_calls*100 > shares, so the rule is also enforced mechanically — but we
design compliant from the start. Compliant convex substitutes are below (Angle C).
Cash-secured short PUTS are allowed (that's the wheel); the rule is about calls/shares.

## Angles (one knob per clone, gen-N discipline)

### A — ITM cash-secured puts  (knob: `put_itm_depth` / conviction-scaled)
Today's wheel sells OTM puts. ITM puts (delta ~0.6–0.8) collect intrinsic + fat
time premium and assign more often → faster accumulation at a **cushioned
effective basis**. Return source = basis discount + premium, NOT extra delta:
at the same target share count you arrive cheaper and with less time-at-risk.
- Scale depth by conviction: deep-cheap z + low IV-rank → go ITM to accumulate
  with cushion; neutral z → stay OTM (current behavior).
- Risk: less convex if it rallies away (keep only premium); assignment ~certain.
- Fill fidelity: grid covers otm −0.2; **wire real_chain for exact ITM strike**.

### B — ITM covered calls for divest/income  (knob: `cc_itm_divest`)
Formalizes the synthetic-early-assignment + hot-shares-divest memories into a
knob. When z rich / surge_z hot / shares flagged for divestment, sell DEEP-ITM
covered calls (delta 0.7–0.9): lock a high-probability called-away exit at a
lower strike while harvesting max premium, vs OTM grind. Pure income on shares
already held — fully covered 1:1. Return-not-exposure by construction.
- Risk: caps upside hard (that's the point — it's the divest path).

### C — compliant convex / "ratio" structures
The covered substitutes for the forbidden naked ratio:
- **C1 call ratio BACKSPREAD (long 2 / short 1):** net LONG gamma/vega, short
  leg covered inside the spread. Pays in vol-EXPANSION / trend-up — the regime
  the grinder misses (see engine_is_grinder). Cheap long convexity partly
  funded by the short. Overlay only in high-IV-rank / momentum-up windows.
- **C2 covered upside-tail ratio (own 200sh, short 2 / long 1 higher call):**
  every short covered by shares; long call caps the upper short's tail. =
  "covered calls with an upside kicker." Compliant.
- **C3 put ratio spread (short 2 cash-secured / long 1):** both shorts
  cash-secured (allowed), long put = defined-risk floor. Accumulate harder
  with a downside backstop.

## Candidate ladder
| key | angle | anchor delta |
|-----|-------|--------------|
| champion_kold15_ivrank_kbh | (anchor) | live champion |
| g11_itmput_conv | A: conviction-scaled ITM put depth | basis/premium |
| g11_itmcc_divest | B: ITM CC on rich-z/hot shares | premium income |
| g11_backspread | C1: call backspread in vol-expansion | convexity |
| g11_covratio | C2: covered upside-tail ratio | upside kicker |
| g11_putratio | C3: cash-secured put ratio | defined-risk accum |
| g11_combo | best of above, stacked | — |

## Fill fidelity (mandatory for this gen)
ITM is where BS-model and the coarse OTM-calibrated grid are LEAST trustworthy.
**Wire real_chain.py (tier-3) into the put/call pricing sites** so ITM legs
price at actual historical bid/ask; fall back to grid only off-chain. This is
the pending tier-3 wiring — gen-11 is the right forcing function.

## Gates (unchanged gauntlet + one addition)
1. Real fills default; open_dte 60; tier-3 real_chain for ITM legs.
2. `audit.py` — covered_calls_only is FATAL (auto-catches any naked ratio);
   plus confound / regime / bootstrap.
3. `trade_forensics.py` — 0 integrity flags; sanity-check assignment rates on
   ITM puts (early-assign haircut already in the cost model).
4. **THE DECISIVE GATE: honest_walkforward sealed OOS — Sharpe ≥ 2.0 OOS AND
   DD that scales proportionately (not super-linear).** Same bar that rejected
   gen-9 and gen-10.
5. Promote only if OOS Sharpe ≥ 2.0 AND OOS return > champion AND audit-clean.

## DECIDED (2026-06-15): build all three compliant ratios; organize gen-11 as a
## DIRECTIONAL-EXPRESSION LIBRARY — each structure expresses a view, the kernel's
## signals (z, IV-rank, surge_z, momentum, backwardation) pick which to deploy.

The unifying frame: gen-11 is not one kernel, it is a *menu of compliant ways to
express a directional thesis*, signal-gated. Map every structure to the view it
expresses and the regime it wants:

### BULLISH expressions (accumulate / lean long)
- ITM cash-secured puts (A) — accumulate at cushioned basis when deep-cheap z.
- Cash-secured put ratio short2/long1 (C3) — aggressive accumulation w/ floor,
  on deep-cheap z + momentum-confirm (respect no-falling-knife).
- Call backspread long2/short1 (C1) — convex upside, for z-cheap + vol-expansion
  / trend-up (the regime the grinder misses). The bullish convexity play.

### NEUTRAL / income expressions
- Covered upside-tail ratio own200/short2/long1 (C2) — extra premium vs 1:1 CC,
  capped upside, when z neutral and IV-rank elevated.
- OTM covered calls (existing grind) — baseline neutral income.

### BEARISH / hedge expressions
- ITM covered calls for divest (B) — monetize + pre-commit exit at premium when
  z rich / surge_z hot / shares flagged hot (hot-shares-divest).
- KOLD book hedge (existing, gen-8) — inverse-ETF hedge of the uncovered book.
- (optional) protective long put overlay — defined-risk tail insurance when
  backwardation-storm / anomaly flags fire (no-falling-knife regime).

### Selection logic (the new kernel brain)
A signal→structure router replaces "always sell OTM put / OTM call":
  deep-cheap z + low IVR + momentum-up  → ITM put or put-ratio or backspread (lean long)
  neutral z + high IVR                  → covered-ratio / OTM CC (harvest)
  rich z / surge hot / hot shares       → ITM CC divest + KOLD (lean defensive)
  anomaly / backwardation storm         → stand down + protective put (no falling knife)

## Candidate ladder (revised) — STATUS
| key | view | structure | status |
|-----|------|-----------|--------|
| champion_kold15_ivrank_kbh | (anchor) | live champion | — |
| g11_itmput_conv | bullish | conviction-scaled ITM put depth | ✅ DONE (OOS-neutral/safe) |
| g11_itmcc_divest | bearish | ITM CC on rich-z/hot shares | ✅ DONE (OOS-neutral/safe) |
| g11_backspread | bullish-convex | call backspread in vol-expansion | ⬜ C1 NEXT |
| g11_covratio | neutral-income | covered upside-tail ratio | ⬜ C2 |
| g11_putratio | bullish | cash-secured put ratio + floor | ⬜ C3 |
| g11_router | all | signal→structure router (best stacked) | ⬜ final |

### Results so far (real fills; OOS = sealed walk-forward, full cost model)
| angle | in-sample (ann/Sh/MDD) | OOS (ann/Sh/MDD) | audit | verdict |
|-------|------------------------|------------------|-------|---------|
| champion (anchor) | +27.3/2.06/-9.6 | +22.1/1.90/-8.7 | — | live |
| A g11_itmput_conv | +28.2/2.09/-9.6 | +22.1/1.90/-8.6 | clean, 1.03x, CI~0 | KEEP-to-stack; OOS-neutral |
| B g11_itmcc_eager | +28.2/2.10/-9.6 | +22.2/1.90/-8.7 | clean, 1.03x, CI~0 | KEEP-to-stack; OOS-neutral |

**Meta-pattern (A+B):** both are compliant + confound-FREE (1.03x shares = geometry
not exposure) and OOS-SAFE, but OOS-NEUTRAL — their triggers (deep-cheap for A,
rich for B) barely fired in the calm/cheap 2024-26 test window. They are
regime-insurance that pays in their target regime, costs nothing otherwise.
The real return upside should come from C1 (backspread convexity in vol-expansion)
and the router (deploy each structure only where its regime lives).

Build order: A (itmput) and B (itmcc) first — smallest deltas off the champion,
reuse existing strike-selection plumbing. Then C1/C2/C3 (need multi-leg support).
Then the router. Each clone = one knob, gated through the full gauntlet, decided
by sealed OOS Sharpe ≥ 2.0 with proportionate DD.
