"""Statistical assignment-probability model for short calls/puts.

Single source of truth for "will this short option get assigned?" — used
by both the live kernel adapter and the backtest replay engine, so live
guidance and backtest decisions stay in sync.

The model has three components:
1. Risk-neutral terminal-ITM probability via BSM N(d2) / N(-d2).
2. Early-assignment kicker: extrinsic-driven. Low extrinsic + deep ITM +
   short DTE → meaningful chance of being assigned before expiry.
3. Decision helpers: expected-value comparison of WAIT vs BTC vs ROLL.

For UNG specifically there's no dividend, so early call exercise is rare
unless the market closes the contract for liquidity. Early put exercise
happens when extrinsic decays to near-zero on deep-ITM puts.
"""
from __future__ import annotations
import math
from typing import Optional


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def assignment_probability(K: float, spot: float, dte: int, iv: float,
                            right: str, premium_market: Optional[float] = None,
                            r: float = 0.045) -> dict:
    """Return a structured P(assignment) for one short option leg.

    Args:
        K: strike
        spot: current underlying
        dte: days to expiry (0 = today)
        iv: implied vol (annualized, e.g., 0.50)
        right: 'CALL' or 'PUT'
        premium_market: current market mid (optional; lets us compute extrinsic)
        r: risk-free

    Returns:
        dict with:
          p_expiry_itm: BSM risk-neutral P(ITM at expiry)
          p_early_kicker: extra prob from extrinsic-driven early-exercise
          p_assign: combined min(0.99, ...)
          intrinsic: max(0, S-K) for call, (K-S) for put — per share
          extrinsic: premium_market - intrinsic (if premium given)
          regime: 'cert' | 'likely' | 'possible' | 'unlikely' | 'remote'
    """
    right = right.upper()
    if dte <= 0:
        # At expiry: deterministic
        if right == 'CALL':
            assigned = spot > K
            intrinsic = max(0.0, spot - K)
        else:
            assigned = spot < K
            intrinsic = max(0.0, K - spot)
        p = 1.0 if assigned else 0.0
        return {
            'p_expiry_itm': p, 'p_early_kicker': 0.0, 'p_assign': p,
            'intrinsic': intrinsic, 'extrinsic': 0.0,
            'regime': 'cert' if assigned else 'remote',
        }
    if spot <= 0 or K <= 0 or iv <= 0:
        return {'p_expiry_itm': 0, 'p_early_kicker': 0, 'p_assign': 0,
                'intrinsic': 0, 'extrinsic': 0, 'regime': 'remote'}

    T = max(1.0, dte) / 365.0
    d1 = (math.log(spot / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if right == 'CALL':
        p_itm = _norm_cdf(d2)
        intrinsic = max(0.0, spot - K)
    else:
        p_itm = _norm_cdf(-d2)
        intrinsic = max(0.0, K - spot)

    # Early-assignment kicker. Logic:
    #   - If we know the market premium, extrinsic = premium - intrinsic.
    #     extrinsic < $0.05/share AND intrinsic > 0 → very near parity → high
    #     chance of early exercise (counterparty captures intrinsic).
    #   - Without premium_market, fall back to "deep ITM + short DTE" heuristic.
    p_early_kicker = 0.0
    if premium_market is not None:
        extrinsic = max(0.0, premium_market - intrinsic)
        if intrinsic > 0:
            if extrinsic < 0.05:
                p_early_kicker = 0.30 if dte <= 5 else 0.15 if dte <= 14 else 0.05
            elif extrinsic < 0.15:
                p_early_kicker = 0.10 if dte <= 5 else 0.05
    else:
        extrinsic = 0.0
        # Fallback heuristic
        if intrinsic > 0:
            itm_pct = intrinsic / spot * 100
            if itm_pct > 5 and dte <= 5:
                p_early_kicker = 0.25
            elif itm_pct > 3 and dte <= 7:
                p_early_kicker = 0.15
            elif itm_pct > 5 and dte <= 14:
                p_early_kicker = 0.08

    p_assign = min(0.99, p_itm + p_early_kicker)

    if p_assign >= 0.85:    regime = 'cert'
    elif p_assign >= 0.55:  regime = 'likely'
    elif p_assign >= 0.25:  regime = 'possible'
    elif p_assign >= 0.05:  regime = 'unlikely'
    else:                   regime = 'remote'

    return {
        'p_expiry_itm': round(p_itm, 4),
        'p_early_kicker': round(p_early_kicker, 4),
        'p_assign': round(p_assign, 4),
        'intrinsic': round(intrinsic, 4),
        'extrinsic': round(extrinsic, 4),
        'regime': regime,
    }


def expected_value_wait_vs_btc(K: float, spot: float, dte: int, iv: float,
                                right: str, entry_prem: float,
                                premium_market: float, contracts: int = 1) -> dict:
    """Compare expected P&L of WAITing (let assignment/expiry resolve naturally)
    vs BTC (close now at market) for a short leg.

    Returns dict with both EVs (per contract, dollars) and a recommendation.

    Wait EV (short call):
      = entry_prem * 100                       (premium kept regardless)
        - p_assign * (E[S_T | ITM] - K) * 100  (assignment loss)
    Wait EV (short put):
      = entry_prem * 100
        - p_assign * (K - E[S_T | ITM]) * 100

    Conditional means use Black-Scholes truncated expectation.

    BTC EV:
      = (entry_prem - premium_market) * 100    (lock realized; pay extrinsic now)
    """
    a = assignment_probability(K, spot, dte, iv, right, premium_market)
    p = a['p_assign']

    if dte <= 0 or iv <= 0 or spot <= 0:
        return {'ev_wait': 0, 'ev_btc': 0, 'recommend': 'wait',
                'note': 'at-expiry or invalid inputs', 'p_assign': p}

    T = max(1, dte) / 365.0
    # Truncated expected spot conditional on terminal ITM, log-normal
    # E[S_T | S_T > K] under risk-neutral with no dividend yield:
    #   = S0 * N(d1) / N(d2)
    # For short PUT: E[S_T | S_T < K] = S0 * N(-d1) / N(-d2)
    d1 = (math.log(spot/K) + (0.045 + 0.5*iv*iv)*T) / (iv*math.sqrt(T))
    d2 = d1 - iv*math.sqrt(T)
    try:
        if right.upper() == 'CALL':
            n_d2 = _norm_cdf(d2)
            cond_e = spot * _norm_cdf(d1) / n_d2 if n_d2 > 1e-6 else K
            loss_per_share_if_assigned = max(0, cond_e - K)
        else:
            n_md2 = _norm_cdf(-d2)
            cond_e = spot * _norm_cdf(-d1) / n_md2 if n_md2 > 1e-6 else K
            loss_per_share_if_assigned = max(0, K - cond_e)
    except Exception:
        loss_per_share_if_assigned = a['intrinsic']

    ev_wait_per_contract = entry_prem * 100 - p * loss_per_share_if_assigned * 100
    ev_btc_per_contract = (entry_prem - premium_market) * 100

    diff = ev_wait_per_contract - ev_btc_per_contract  # positive → wait better
    recommend = 'wait' if diff > 0 else 'btc'

    return {
        'ev_wait': round(ev_wait_per_contract * contracts, 2),
        'ev_btc': round(ev_btc_per_contract * contracts, 2),
        'ev_diff_wait_minus_btc': round(diff * contracts, 2),
        'recommend': recommend,
        'p_assign': p,
        'p_regime': a['regime'],
        'cond_e_if_itm': round(cond_e, 4) if 'cond_e' in dir() else None,
        'loss_if_assigned_per_share': round(loss_per_share_if_assigned, 4),
    }


def _self_test():
    # Sanity: 1 DTE deeply ITM call, no extrinsic — wait better (assignment free)
    a = assignment_probability(K=11.50, spot=12.0, dte=1, iv=0.40, right='CALL',
                                premium_market=0.55)
    print('1d ITM call:', a)
    # ATM 7 DTE put — possible but not certain
    a = assignment_probability(K=12.0, spot=12.0, dte=7, iv=0.40, right='PUT',
                                premium_market=0.30)
    print('7d ATM put:', a)
    # Deep OTM 30d put — remote
    a = assignment_probability(K=10.0, spot=12.0, dte=30, iv=0.40, right='PUT',
                                premium_market=0.10)
    print('30d OTM put:', a)


if __name__ == '__main__':
    _self_test()
