"""Historical IV model for UNG.

Real historical chains aren't free. This module produces a time-varying
IV estimate per (date, strike, dte, right) that's MUCH more realistic
than a fixed 0.55:

  IV(t, K, T, right) = base_iv(t) * term_adj(T) * skew_adj(K/S, right)

Where:
  base_iv(t)   = blend(realized_vol_30d, VIX-scaled regime) + vol risk premium
  term_adj(T)  = mild contango / backwardation by DTE
  skew_adj()   = empirical UNG put-skew (~10% put premium vs call equivalent)

Calibration anchors:
  - Realized UNG vol historically ~40-80% (calm) to 120%+ (2022)
  - Vol risk premium: implied ~ realized * 1.1 on average
  - UNG put skew: puts trade richer by ~5-15% IV
"""
import math
import numpy as np
import pandas as pd


VOL_RISK_PREMIUM = 1.10   # IV typically 10% higher than realized
PUT_SKEW = 0.10           # OTM puts trade ~10% higher IV
TERM_SLOPE = 0.02         # +2% IV per +30 DTE (mild contango)
VIX_ANCHOR = 20.0
VIX_BETA = 0.015          # UNG IV gets +1.5pp per VIX point above 20
MIN_IV = 0.25
MAX_IV = 2.50


def precompute_realized_vol(df: pd.DataFrame, col: str = 'UNG') -> pd.DataFrame:
    """Add rolling realized vol columns to df."""
    out = df.copy()
    if col not in out.columns:
        return out
    rets = (out[col] / out[col].shift(1)).apply(np.log)
    out['rv_10'] = rets.rolling(10).std() * math.sqrt(252)
    out['rv_30'] = rets.rolling(30).std() * math.sqrt(252)
    out['rv_90'] = rets.rolling(90).std() * math.sqrt(252)
    return out


def base_iv_for_date(row) -> float:
    """Base ATM IV for the given snapshot row.

    Blends realized vol (recent + medium) with VIX regime adjustment.
    """
    rv_10 = row.get('rv_10')
    rv_30 = row.get('rv_30')
    rv_90 = row.get('rv_90')
    vix = row.get('VIX')

    # Use whichever realized vols are available
    rvs = [v for v in (rv_10, rv_30, rv_90) if v and not pd.isna(v) and v > 0]
    if not rvs:
        rv = 0.55  # fallback
    else:
        # Weight recent more, but blend medium/long for stability
        if len(rvs) == 3:
            rv = 0.5 * rvs[0] + 0.3 * rvs[1] + 0.2 * rvs[2]
        else:
            rv = float(np.mean(rvs))

    iv = rv * VOL_RISK_PREMIUM

    # VIX regime: each VIX point above 20 adds 1.5pp to UNG IV
    if vix and not pd.isna(vix):
        iv += (vix - VIX_ANCHOR) * VIX_BETA

    return max(MIN_IV, min(MAX_IV, iv))


def adjust_iv(base: float, strike: float, spot: float, dte: int, right: str) -> float:
    """Adjust base IV for moneyness/skew/term structure."""
    if spot <= 0:
        return base
    iv = base
    # Term structure: linear add per 30 DTE
    iv += (dte / 30.0 - 1.0) * TERM_SLOPE
    # Put skew: OTM puts richer
    moneyness = strike / spot
    if right == 'P':
        if moneyness < 1.0:  # OTM put
            otm_pct = 1.0 - moneyness
            iv += PUT_SKEW * otm_pct * 5  # max ~+5pp at deep OTM
    elif right == 'C':
        if moneyness > 1.0:  # OTM call
            otm_pct = moneyness - 1.0
            iv -= 0.05 * otm_pct * 3  # mild reverse skew on calls
    return max(MIN_IV, min(MAX_IV, iv))


def iv_for_quote(row, strike: float, spot: float, dte: int, right: str) -> float:
    """One-shot: get IV for a specific quote from a row."""
    base = base_iv_for_date(row)
    return adjust_iv(base, strike, spot, dte, right)


if __name__ == '__main__':
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, 'cache/master_dataset.csv'),
                     parse_dates=['Date'], index_col=0).dropna(subset=['UNG'])
    df = precompute_realized_vol(df)

    print("=== Calibrated IV Model — Sanity Check ===")
    print(f"Rows: {len(df)}")
    print()
    print(f"{'Date':<12} {'UNG':>7} {'VIX':>6} {'rv_30':>7} {'base_iv':>8} {'ATM_30d':>9} {'P_10%OTM':>9}")
    sample_dates = ['2021-06-15', '2021-12-15', '2022-03-15', '2022-08-15',
                    '2023-02-15', '2024-01-15', '2025-01-15', '2026-05-29']
    for d in sample_dates:
        try:
            row = df.loc[d:d].iloc[0]
        except (IndexError, KeyError):
            continue
        spot = float(row.get('UNG') or 0)
        vix = float(row.get('VIX') or float('nan'))
        rv30 = float(row.get('rv_30') or float('nan'))
        base = base_iv_for_date(row)
        atm = adjust_iv(base, spot, spot, 30, 'C')
        K_otm_p = round(spot * 0.90, 2)
        iv_otm_p = adjust_iv(base, K_otm_p, spot, 30, 'P')
        print(f"{d:<12} {spot:>7.2f} {vix:>6.1f} {rv30:>7.2%} {base:>8.2%} {atm:>9.2%} {iv_otm_p:>9.2%}")
