"""STAGE 1+2 (revised: probability-weighted, DTE-aware). Fit a Z-conditional model of UNG's drift+vol,
then P(a short put assigns WITHIN ITS OWN DTE | Z) = N((ln(K/S) − μ(Z)·dte)/(σ·√dte)). The EXPECTED
assignment-delta = Σ P(assign)·qty·100 — weighted by what's actually likely in the contract's life
(time matters: a 7-DTE OTM put barely counts; a 45-DTE near-money put counts a lot). Outputs μ/σ
coefficients for the engine + the current book's expected assignment-delta.

  venv/bin/python research/spy_vol/ung_scenario_delta.py
"""
import os
import sys
import math
import json
import urllib.request
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(THIS, '..', '..', 'backtest')
sys.path.insert(0, BT)
from replay_engine import precompute_factor_z, compute_historical_z   # noqa: E402


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def p_assign(K, S, dte, a, b, z, sig):
    """P(UNG_{t+dte} < K | Z=z) — probability the short put is ITM at expiry. DTE-aware (σ·√dte)."""
    if dte <= 0 or S <= 0:
        return 1.0 if S < K else 0.0
    mu = (a + b * z) * dte
    return N((math.log(K / S) - mu) / (sig * math.sqrt(dte)))


def live_puts():
    """current short puts as {(K, dte): qty} from the live book; fallback to known snapshot."""
    try:
        d = json.loads(urllib.request.urlopen('http://127.0.0.1:10001/api/live', timeout=40).read())
        spot = float(d.get('spot') or 11.6)
        shares = int(d.get('coverage', {}).get('shares') or 3400)
        today = pd.Timestamp.today().normalize()
        out = {}
        for c in (d.get('concentration') or []):
            if c.get('right') != 'PUT':
                continue
            K = float(c['strike']); n = int(c['contracts'])
            exps = c.get('expiries') or []
            per = n // max(len(exps), 1)
            for e in exps:
                dte = max(1, (pd.Timestamp(e).normalize() - today).days)
                out[(K, dte)] = out.get((K, dte), 0) + per
            if not exps:
                out[(K, 25)] = out.get((K, 25), 0) + n
        return shares, spot, out
    except Exception:
        return 3400, 11.61, {(11.0, 16): 8, (11.0, 23): 9, (11.0, 30): 8, (10.5, 23): 1}


def main():
    df = pd.read_csv(os.path.join(BT, 'cache', 'master_dataset.csv'), index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    ung = df['UNG']
    lr = np.log(ung / ung.shift(1))
    z = pd.Series([compute_historical_z(df.iloc[i]) for i in range(len(df))], index=df.index)
    d = pd.DataFrame({'lr': lr, 'z': z}).dropna()
    # fit daily drift μ(z) = a + b·z  (high z = bearish storage → lower/negative drift)
    b, a = np.polyfit(d['z'].values, d['lr'].values, 1)
    sig = float(d['lr'].std())
    print(f"=== Z-conditional UNG model (daily, 2021-2026) ===")
    print(f"  drift μ(z) = {a:+.5f} {b:+.5f}·z   |   σ_daily = {sig:.4f}  ({sig*math.sqrt(252):.0%} annual)")
    print(f"  → at z=0 drift {a*252:+.0%}/yr; at z=+1 {(a+b)*252:+.0%}/yr; at z=-1 {(a-b)*252:+.0%}/yr\n")
    # validate parametric vs empirical P(assign) at a couple (dte, moneyness) points
    print("  validate P(assign) param vs empirical (z near 0):")
    znear = d[abs(d['z']) < 0.3]
    for dte, mny in [(10, -0.05), (21, -0.05), (42, -0.05), (21, 0.0)]:
        K_S = 1 + mny
        idx = [i for i in range(len(df) - dte) if abs(z.iloc[i]) < 0.3]
        emp = np.mean([ung.iloc[i + dte] / ung.iloc[i] - 1 < mny for i in idx]) if idx else float('nan')
        par = p_assign(K_S, 1.0, dte, a, b, 0.0, sig)
        print(f"    dte {dte:>2} K/S {K_S:.2f}: param {par:.0%}  empirical {emp:.0%}")

    shares, spot, puts = live_puts()
    zc = float(z.iloc[-1])
    print(f"\n=== CURRENT BOOK: z={zc:+.2f}  spot ${spot:.2f}  shares {shares} ===")
    print(f"  {'put':<16}{'dte':>5}{'P(assign)':>11}{'E[assign Δ]':>13}")
    tot = 0.0
    for (K, dte), q in sorted(puts.items()):
        p = p_assign(K, spot, dte, a, b, zc, sig)
        eΔ = p * q * 100
        tot += eΔ
        print(f"  {q}× ${K:<5} {dte:>4}d{p:>10.0%}{eΔ:>+13.0f}")
    print(f"  {'TOTAL expected assignment Δ':<34}{tot:>+13.0f}  (= {tot/shares*100:.0f}% of your {shares} shares)")
    print(f"  book Δ now ≈ {shares} (shares) + small option Δ; a realistic outcome ADDS ~{tot:+.0f} via assignment.")

    # cap implication: how many $11 puts keep E[assign Δ] within target (e.g. 35% of shares)
    target = 0.35 * shares
    # use the nearest $11 dte present
    dte11 = next((dte for (K, dte) in puts if K == 11.0), 23)
    per = p_assign(11.0, spot, dte11, a, b, zc, sig) * 100
    cap = int(target / per) if per > 0 else 999
    held11 = sum(q for (K, dte), q in puts.items() if K == 11.0)
    print(f"\n  GAMMA-AWARE CAP (E[assign Δ] ≤ {target:.0f} = 35% of shares):")
    print(f"    $11 put ({dte11}d) adds {per:.1f} expected-Δ each at z={zc:+.2f} → cap ≈ {cap} contracts (you hold {held11})")
    print(f"\nENGINE COEFFS:  scenario_mu_a={a:.6f}  scenario_mu_b={b:.6f}  scenario_sigma={sig:.5f}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
