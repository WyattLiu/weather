"""Beam-search engine — backtest port of production's recommend cycle.

Mirrors ung_visualizer.py architecture:

  candidates = generate_candidates(portfolio_state, spot, iv, today)
  scored     = [_eval_candidate(state, c, spot, iv, today, q0) for c in candidates]
  best       = beam_top_k(scored, k=BEAM_WIDTH)
  apply(best)

Production has 21 candidate types and a $-normalized multi-objective
quality function. This module implements the structure and a SUBSET of
kernels — incrementally extending toward parity.

Status: SKELETON. Currently implements:
  OPEN_PUT, OPEN_CC (covered call), TAKE_PROFIT_PUT, TAKE_PROFIT_CALL,
  ROLL_DOWN_PUT, ROLL_UP_CALL, CLOSE_CC_ELEVATOR, BUY_BOXX, HOLD

TODO (port from production):
  - ASSIGNMENT (cash-secured put landing)
  - LET_EXPIRE
  - ADD (stack same strike/expiry)
  - BUY_PUT (tail hedge)
  - SELL_SHARES, BUY_SHARES
  - WAITING / multi-step beam chaining
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Any, Callable, Optional
import copy


BEAM_WIDTH = 8
DTE_LADDER = [7, 14, 30, 45, 60]   # candidate DTEs
OTM_PUT_LADDER = [0.05, 0.10, 0.15, 0.20]
OTM_CALL_LADDER = [0.0, 0.03, 0.05, 0.10]
TP_THRESHOLD = 0.50   # close shorts at 50% profit


# ══════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════

@dataclass
class ShortLeg:
    entry: date
    K: float
    dte: int
    qty: int
    entry_prem: float
    right: str         # 'P' or 'C'
    tag: str = ''      # original kernel name


@dataclass
class LongLeg:
    entry: date
    K: float
    dte: int
    qty: int
    cost: float
    right: str
    tag: str = ''


@dataclass
class BeamState:
    today: date
    spot: float
    iv_fn: Callable[[float, int, str], float]
    cash: float
    shares: int
    boxx: int
    short_legs: List[ShortLeg] = field(default_factory=list)
    long_legs: List[LongLeg] = field(default_factory=list)
    # Quality cache
    quality: float = 0.0

    def nav(self, mark_options: bool = True) -> float:
        n = self.cash + self.shares * self.spot + self.boxx * 117
        if mark_options:
            for leg in self.short_legs:
                T = max(1, leg.dte) / 365
                from replay_engine import bs_put, bs_call  # type: ignore
                f = bs_put if leg.right == 'P' else bs_call
                iv = self.iv_fn(leg.K, leg.dte, leg.right)
                # Short = negative value
                n -= f(self.spot, leg.K, T, iv) * 100 * leg.qty
            for leg in self.long_legs:
                T = max(1, leg.dte) / 365
                from replay_engine import bs_put, bs_call  # type: ignore
                f = bs_put if leg.right == 'P' else bs_call
                iv = self.iv_fn(leg.K, leg.dte, leg.right)
                n += f(self.spot, leg.K, T, iv) * 100 * leg.qty
        return float(n)


# ══════════════════════════════════════════════════════════
# Candidate generation
# ══════════════════════════════════════════════════════════

def generate_candidates(state: BeamState, regime_z: float) -> List[Dict[str, Any]]:
    """Produce all candidate kernels for the current state.

    Each candidate is a dict: {type, strike, dte, qty, right, est_premium, ...}
    """
    spot = state.spot
    cands: List[Dict[str, Any]] = []

    # ---- TAKE PROFIT (close shorts at 50% gain) ----
    from replay_engine import bs_put, bs_call  # type: ignore
    for leg in state.short_legs:
        T = max(1, leg.dte) / 365
        f = bs_put if leg.right == 'P' else bs_call
        iv = state.iv_fn(leg.K, leg.dte, leg.right)
        cv = f(spot, leg.K, T, iv)
        if cv < leg.entry_prem * TP_THRESHOLD:
            cands.append({
                'type': 'TP_' + leg.right, 'leg': leg,
                'close_premium': cv,
                'realized': (leg.entry_prem - cv) * 100 * leg.qty,
            })

    # ---- ROLL DOWN PUT (when ITM and >5d to expiry) ----
    for leg in state.short_legs:
        if leg.right != 'P':
            continue
        if leg.dte <= 5:
            continue
        if spot >= leg.K * 0.98:
            continue
        T = max(1, leg.dte) / 365
        iv = state.iv_fn(leg.K, leg.dte, leg.right)
        cv = bs_put(spot, leg.K, T, iv)
        new_K = round(spot * 0.90)
        new_prem = bs_put(spot, new_K, 30/365, state.iv_fn(new_K, 30, 'P'))
        cands.append({
            'type': 'ROLL_DOWN_P', 'leg': leg,
            'close_premium': cv, 'new_K': new_K,
            'new_dte': 30, 'new_prem': new_prem,
        })

    # ---- ROLL UP CALL (when ITM, <=7d, mild bullish regime) ----
    for leg in state.short_legs:
        if leg.right != 'C' or leg.dte > 7 or spot <= leg.K:
            continue
        if regime_z < -0.25:
            continue
        T = max(1, leg.dte) / 365
        iv = state.iv_fn(leg.K, leg.dte, leg.right)
        cv = bs_call(spot, leg.K, T, iv)
        new_K = round(spot * 1.05)
        new_prem = bs_call(spot, new_K, 30/365, state.iv_fn(new_K, 30, 'C'))
        if new_prem < 0.05:
            continue
        cands.append({
            'type': 'ROLL_UP_C', 'leg': leg,
            'close_premium': cv, 'new_K': new_K,
            'new_dte': 30, 'new_prem': new_prem,
        })

    # ---- ELEVATOR CLOSE (deep ITM call, low extrinsic, at peak) ----
    # (Same conditions as in replay_engine — needs price spike + near 60d high)
    # Skipped in skeleton; pass-through via candidate generator extension.

    # ---- OPEN PUT (ladder over OTM% × DTE) ----
    for otm in OTM_PUT_LADDER:
        for dte in DTE_LADDER:
            K = round(spot * (1 - otm))
            T = dte / 365
            iv = state.iv_fn(K, dte, 'P')
            prem = bs_put(spot, K, T, iv)
            if prem < 0.05:
                continue
            # Sizing: 5 contracts default
            cands.append({
                'type': 'OPEN_P', 'K': K, 'dte': dte, 'right': 'P',
                'qty': 5, 'premium': prem,
            })

    # ---- OPEN COVERED CALL (need shares) ----
    if state.shares >= 100:
        max_qty = state.shares // 100
        for otm in OTM_CALL_LADDER:
            for dte in DTE_LADDER:
                K = round(spot * (1 + otm))
                T = dte / 365
                iv = state.iv_fn(K, dte, 'C')
                prem = bs_call(spot, K, T, iv)
                if prem < 0.05:
                    continue
                cands.append({
                    'type': 'OPEN_C', 'K': K, 'dte': dte, 'right': 'C',
                    'qty': min(5, max_qty), 'premium': prem,
                })

    # ---- BUY BOXX (cash → 4% yield) ----
    excess_cash = state.cash - 20_000
    if excess_cash > 5_000:
        qty = int(excess_cash * 0.6 / 117)
        if qty >= 10:
            cands.append({'type': 'BUY_BOXX', 'qty': qty})

    # ---- HOLD (always available) ----
    cands.append({'type': 'HOLD'})

    return cands


# ══════════════════════════════════════════════════════════
# Apply candidate → new state
# ══════════════════════════════════════════════════════════

def apply_candidate(state: BeamState, c: Dict[str, Any]) -> BeamState:
    """Apply a candidate to the state — pure function returning a new state.
    Skeleton implementation; production has more sophisticated transitions."""
    new = copy.deepcopy(state)
    t = c['type']

    if t in ('TP_P', 'TP_C'):
        leg = c['leg']
        new.short_legs = [l for l in new.short_legs if l is not leg]
        new.cash += c['realized']

    elif t == 'ROLL_DOWN_P':
        leg = c['leg']
        new.short_legs = [l for l in new.short_legs if l is not leg]
        new.cash -= c['close_premium'] * 100 * leg.qty
        new.cash += c['new_prem'] * 100 * leg.qty
        new.short_legs.append(ShortLeg(
            entry=new.today, K=c['new_K'], dte=c['new_dte'],
            qty=leg.qty, entry_prem=c['new_prem'], right='P',
            tag='ROLL_DOWN',
        ))

    elif t == 'ROLL_UP_C':
        leg = c['leg']
        new.short_legs = [l for l in new.short_legs if l is not leg]
        new.cash -= c['close_premium'] * 100 * leg.qty
        new.cash += c['new_prem'] * 100 * leg.qty
        new.short_legs.append(ShortLeg(
            entry=new.today, K=c['new_K'], dte=c['new_dte'],
            qty=leg.qty, entry_prem=c['new_prem'], right='C',
            tag='ROLL_UP',
        ))

    elif t == 'OPEN_P':
        new.cash += c['premium'] * 100 * c['qty']
        new.short_legs.append(ShortLeg(
            entry=new.today, K=c['K'], dte=c['dte'],
            qty=c['qty'], entry_prem=c['premium'], right='P', tag='OPEN_P',
        ))

    elif t == 'OPEN_C':
        new.cash += c['premium'] * 100 * c['qty']
        new.short_legs.append(ShortLeg(
            entry=new.today, K=c['K'], dte=c['dte'],
            qty=c['qty'], entry_prem=c['premium'], right='C', tag='OPEN_C',
        ))

    elif t == 'BUY_BOXX':
        new.cash -= c['qty'] * 117
        new.boxx += c['qty']

    return new


# ══════════════════════════════════════════════════════════
# Evaluator (simplified port of evaluate_portfolio_quality)
# ══════════════════════════════════════════════════════════

def evaluate_quality(state: BeamState, regime_z: float,
                     target_weekly_income: float = 1500.0) -> float:
    """Quality scalar — higher = better.

    Components ($-normalized, mirrors production structure but simplified):
      + weekly income (theta proxy from short option premiums / weeks_held)
      - delta gap (squared shortfall from target delta)
      - CVaR penalty (rough: short_qty * spot * 0.10 as worst-case down move)
      + regime tilt (cheap = more shorts good; rich = fewer shorts good)
    """
    # Income proxy: sum of short premiums × (1/weeks to expiry on each leg)
    weekly_income = 0.0
    for leg in state.short_legs:
        weeks = max(1, leg.dte / 7)
        weekly_income += leg.entry_prem * 100 * leg.qty / weeks

    income_gap = weekly_income - target_weekly_income
    income_score = income_gap if income_gap >= 0 else income_gap * 1.5

    # Delta: shares + short put delta (≈ -0.5 each, scaled by moneyness)
    # + short call delta (≈ +0.5 each scaled)
    short_p_qty = sum(l.qty for l in state.short_legs if l.right == 'P')
    short_c_qty = sum(l.qty for l in state.short_legs if l.right == 'C')
    # Simplified: shares + 50*short_puts - 50*short_calls
    delta_shares = state.shares + 50 * short_p_qty - 50 * short_c_qty
    # Target delta is a function of regime
    if regime_z > 0.5:
        target_delta = 8000   # cheap → long
    elif regime_z < -0.5:
        target_delta = 4000   # rich → reduce
    else:
        target_delta = 6000
    delta_gap_score = -((delta_shares - target_delta) ** 2) * 0.001

    # CVaR rough: 10% adverse move on short put assignment risk
    cvar_penalty = -short_p_qty * 100 * state.spot * 0.10 * 0.05  # α=5%

    # Regime tilt: cheap = reward more shorts; rich = penalize
    regime_tilt = regime_z * 500

    return float(income_score + delta_gap_score + cvar_penalty + regime_tilt)


# ══════════════════════════════════════════════════════════
# Beam selection
# ══════════════════════════════════════════════════════════

def beam_step(state: BeamState, regime_z: float,
              k: int = BEAM_WIDTH) -> Optional[Dict[str, Any]]:
    """Generate, score, return the best candidate (single-step beam).
    Returns None if HOLD wins."""
    q0 = evaluate_quality(state, regime_z)
    candidates = generate_candidates(state, regime_z)

    scored = []
    for c in candidates:
        if c['type'] == 'HOLD':
            scored.append((0.0, c))
            continue
        try:
            new_state = apply_candidate(state, c)
            q1 = evaluate_quality(new_state, regime_z)
            scored.append((q1 - q0, c))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    if not top or top[0][0] <= 0:
        return None
    return top[0][1]


if __name__ == '__main__':
    # Smoke test
    def iv_fn(_K, _dte, _right):
        return 0.60
    state = BeamState(
        today=date(2024, 1, 15), spot=20.0, iv_fn=iv_fn,
        cash=50_000, shares=6200, boxx=0,
    )
    cands = generate_candidates(state, regime_z=0.6)
    print(f"Generated {len(cands)} candidates")
    for c in cands[:10]:
        print(f"  {c['type']:<12} {c}")
    best = beam_step(state, regime_z=0.6)
    print(f"\nBest pick: {best}")
