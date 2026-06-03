# Validated Kernel → Production Integration Guide

**Goal:** plug backtest-validated logic into `ung_visualizer.py` (port 9999) with **zero regression risk**.

## Architecture: read-only side-car

```
┌────────────────────────────────────┐      ┌───────────────────────────────┐
│ ung_visualizer.py (UNCHANGED)      │      │ validated_kernel_adapter.py   │
│                                    │      │                               │
│ compute_recommendations() ─────────│      │ validated_verdict(spot, pos)  │
│       │                            │      │       │                       │
│       ▼                            │      │       ▼                       │
│ HTML_PAGE ◄──── /api/data          │      │ JSON {regime, target_shares,  │
│                                    │      │       share_delta, recs[],    │
│ NEW: HTML panel ◄──── /api/validated────►│       warnings[], iv_shape}   │
└────────────────────────────────────┘      └───────────────────────────────┘
```

- Production keeps its existing `compute_recommendations()` 100% untouched
- New `/api/validated` endpoint just calls the adapter
- Dashboard renders a side panel ("Backtest Says...") with the verdict
- Production rec engine and validated kernel run **independently**; never cross-mutate

## Three ways to integrate (pick one)

### Option A — Read-only validation panel (recommended, safest)

5-line patch to `ung_visualizer.py`. Adds a NEW route, no existing code touched.

**Top of file** (near other imports, ~line 35):
```python
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'weather',
                                    'ibkr_guided_trade', 'backtest'))
    from validated_kernel_adapter import validated_verdict
except Exception:
    validated_verdict = None
```

**Inside `Handler.do_GET`** (~line 10100, after existing routes):
```python
elif parsed.path == '/api/validated':
    if validated_verdict is None:
        data = {'available': False, 'reason': 'adapter not loaded'}
    else:
        try:
            data = validated_verdict(UNG_PRICE, WS_POSITIONS or [])
        except Exception as e:
            data = {'available': False, 'reason': str(e)}
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.end_headers()
    self.wfile.write(json.dumps(data, default=str).encode('utf-8'))
```

**Dashboard JS** (in HTML_PAGE): add an `fetch('/api/validated')` call and render the resulting `recommendations[]` + `warnings[]` into a side panel. Zero impact on existing dashboard logic.

**Regression risk: zero.** Adapter is read-only; production rec flow is untouched. Adapter failure shows "not available" — no cascade.

### Option B — A/B compare prod recs vs validated recs

Same patch as A, plus a small dashboard table that shows production's top rec next to the validated kernel's verdict. Helps you spot disagreements without acting on them.

**Regression risk: zero.** Pure display.

### Option C — Feature-flagged behavior swap

For specific mechanics (e.g., the `dd_trim_trigger`, the `z_share_target` sizing), gate the swap behind a config flag in production:

```python
USE_VALIDATED_Z_TARGET = False  # flip to True after parallel testing
```

Then in `compute_recommendations`, branch on the flag. This IS a behavior change — only enable after running parallel for 2-4 weeks and confirming the validated suggestions land where you'd want.

**Regression risk: real.** Defer until A/B has run long enough to trust.

## What the adapter returns (validated against your WS state)

For your current 5400 shares + 14 short puts + 14 short calls (NEUTRAL z=+0.20):

```
recommendations:
  - BUY 800 UNG shares  (priority: high)
    why: z=+0.20 → NEUTRAL → target 6200
  - Standard 5% OTM CCs on uncovered shares (K ≈ $12.09)
    why: NEUTRAL regime; normal premium harvest mode

warnings:
  - Walk-forward worst 12mo MDD: -17% (sample-biased -7%)

iv_shape today:
  atm_iv: 0.47, put_skew: -0.01, call_skew: +0.01, term_slope: 0.0
```

## Validation benefits this gives you

1. **Real-time backtest verdict alongside production rec** — see if both agree on direction
2. **Disagreement = signal**: if production says "sell ITM CC" but validated says "buy shares", you've found either a production miscalibration OR a backtest miss; both worth investigating
3. **Walk-forward MDD warning surfaced live** — production currently doesn't show the realistic -17% worst case
4. **Risk metric: put_collateral_pct_nav** — flags when you're approaching over-leverage (your current value: 22%, healthy)
5. **IV shape live readout** — currently production uses calibrated IV; this surfaces real PG-backed surface signal

## How to roll out safely

Week 0: Apply Option A patch. Observe `/api/validated` JSON in browser. No dashboard change.

Week 1: Add the side panel. Watch for daily agreement/disagreement with production recs.

Week 2-4: Log disagreements. Investigate top 10 to understand if validated is better or worse for your specific state.

Week 4+: If validated is consistently better in N cases, consider Option C for those specific mechanics. Otherwise keep as side panel forever.

## What's NOT plugged in

- Order routing: validated kernel never calls WS to place orders
- Position state mutation: it reads positions, never writes
- IV ingestion: the adapter uses backtest's PG IV surface; production keeps its own IV chain pull. Two independent IV sources is FEATURE — disagreement is a quality signal

## Files

- `backtest/validated_kernel_adapter.py` — the adapter (200 lines)
- `backtest/INTEGRATION_GUIDE.md` — this file
- `backtest/replay_engine.py:STRATEGIES['champion_target_25_dd_trim']` — the validated config
