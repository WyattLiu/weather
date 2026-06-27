"""Ornstein-Uhlenbeck mean-reversion model for UNG — the elegant short-horizon process.

  dX = θ(μ − X)dt + σ dW    (X = log-price)

Fit on a ROLLING window so μ tracks the LOCAL level (UNG's secular trend is handled by the
slow regime layer; OU captures the fast pull-to-mean around it). Gives two things:

  • OU z = (logS − μ_local) / σ_stationary   → a SHORT-scale 'cheap/rich vs the local mean'
    signal that NATURALLY drives small buy-low / sell-high (small near μ, larger at extremes).
  • forward conditional distribution  E[X_{t+h}] = μ + (X_t−μ)e^{−θh},
    Var = σ²/(2θ)(1−e^{−2θh})   → Monte-Carlo what-if (P(assignment), E[P&L], CVaR) that
    respects mean reversion, unlike a static ±2σ kernel.

  venv/bin/python backtest/ou_model.py     # calibrate + validate (half-life, reversion edge)
"""
import os
import numpy as np
import pandas as pd


def fit_ou(logx):
    """Fit OU via AR(1) OLS on daily log-prices. dt = 1 day. Returns dict or None."""
    x = np.asarray(logx, float)
    if len(x) < 20:
        return None
    x0, x1 = x[:-1], x[1:]
    b, a = np.polyfit(x0, x1, 1)            # x1 = b·x0 + a
    if not (0 < b < 1):
        return None
    theta = 1.0 - b                          # per-day mean-reversion speed
    mu = a / (1.0 - b)
    resid = x1 - (b * x0 + a)
    sig = resid.std(ddof=2)
    stat_sd = sig / math_sqrt(1.0 - b * b)   # stationary std of the level
    half_life = math_log(2.0) / theta        # days to revert halfway
    return {'theta': theta, 'mu': mu, 'sigma': sig, 'stat_sd': stat_sd,
            'half_life': half_life}


def math_sqrt(x):
    return float(np.sqrt(max(x, 1e-12)))


def math_log(x):
    return float(np.log(max(x, 1e-12)))


def ou_z_series(prices, window=90):
    """Rolling OU z = (logS − μ_local)/σ_stationary. Causal (uses trailing window only)."""
    lp = np.log(np.asarray(prices, float))
    z = np.full(len(lp), np.nan)
    for i in range(window, len(lp)):
        f = fit_ou(lp[i - window:i])
        if f and f['stat_sd'] > 1e-6:
            z[i] = (lp[i] - f['mu']) / f['stat_sd']
    return pd.Series(z, index=getattr(prices, 'index', None))


def forward_moments(logS, params, h_days):
    """Conditional (mean, std) of log-price h days ahead under OU."""
    th, mu, sig = params['theta'], params['mu'], params['sigma']
    e = math_exp(-th * h_days)
    mean = mu + (logS - mu) * e
    var = (sig * sig) / (2 * th) * (1 - math_exp(-2 * th * h_days)) if th > 0 else sig * sig * h_days
    return mean, math_sqrt(var)


def math_exp(x):
    return float(np.exp(np.clip(x, -50, 50)))


def prob_below(S, K, params, h_days):
    """P(spot < K at horizon h) under OU — for what-if / assignment probability."""
    from scipy.stats import norm
    mean, sd = forward_moments(math_log(S), params, h_days)
    return float(norm.cdf((math_log(K) - mean) / max(sd, 1e-6)))


def validate(start='2021-06-17', end='2026-06-16', window=90):
    THIS = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(THIS, 'cache', 'master_dataset.csv'), index_col=0, parse_dates=True)
    u = df['UNG'].dropna(); u = u[~u.index.duplicated()].loc[start:end]
    # global fit (for half-life sanity) + rolling z
    g = fit_ou(np.log(u.values))
    z = ou_z_series(u, window)
    print(f"=== OU MEAN-REVERSION MODEL (UNG, {start}→{end}) ===")
    print(f"  global half-life: {g['half_life']:.0f} days  (θ={g['theta']:.4f}/day, σ={g['sigma']:.4f}/day)")
    print(f"  rolling-{window}d OU z: range {np.nanmin(z):.2f}..{np.nanmax(z):.2f}, |z|>1.5 on {np.mean(np.abs(z)>1.5)*100:.0f}% of days")
    # VALIDATION: does OU z predict short-horizon MEAN REVERSION? (cheap→up, rich→down)
    print("\n  forward return by OU-z bucket (the buy-low/sell-high edge):")
    for h in (5, 10, 21):
        fwd = (u.shift(-h) / u - 1) * 100
        al = pd.concat([z.rename('z'), fwd.rename('f')], axis=1).dropna()
        lo = al[al.z < -1.0]['f']; hi = al[al.z > 1.0]['f']
        print(f"    fwd-{h:2}d: z<-1 (cheap) {lo.mean():+5.2f}%   z>+1 (rich) {hi.mean():+5.2f}%   "
              f"spread {lo.mean()-hi.mean():+5.2f}pp   corr {al.z.corr(al.f):+.2f}")
    print("\n  → negative corr / cheap>rich = mean reversion works; |z| sizes the buy-low/sell-high amount.")


if __name__ == '__main__':
    validate()
