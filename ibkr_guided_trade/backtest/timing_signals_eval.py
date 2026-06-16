"""Directional-timing signal evaluation for UNG — does any well-known method
actually predict forward returns? Isolated from the options kernel so we measure
DIRECTIONAL edge directly (IC, conditional fwd returns, long/flat Sharpe, t-stat).

Each signal is expressed as a z-scored "expected UNG direction" (positive=bullish)
so ICs are comparable. We measure forward 5d / 21d / 63d UNG returns.

Honest-stats notes:
- forward returns are computed with NO lookahead (signal at t, return t->t+h).
- overlapping h-day returns autocorrelate → longer-horizon t-stats are optimistic;
  we also report a long/flat daily-rebalanced backtest (non-overlapping P&L) as the
  reality check, plus a block-bootstrap p-value on the 21d IC.
"""
import os
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS, 'cache')


def _z(s, win=252):
    m = s.rolling(win, min_periods=60).mean()
    sd = s.rolling(win, min_periods=60).std()
    return (s - m) / sd.replace(0, np.nan)


def _rsi(price, n=14):
    d = price.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_signals(df):
    p = df['UNG'].astype(float)
    ng = df['NG'].astype(float)
    sig = pd.DataFrame(index=df.index)

    # 1. BACKWARDATION (term structure): HH spot - futures. Memory: backwardation
    #    > +$0.40 precedes UNG weakness (spot crashes back to futures). BEARISH when
    #    high → bullish-direction signal = NEGATIVE basis-z.
    if 'eia_hh_spot_daily' in df.columns:
        basis = (df['eia_hh_spot_daily'].astype(float) - ng).ffill(limit=5)
        sig['backwardation'] = -_z(basis)        # high basis -> bearish

    # 2. TREND: 50/200 MA crossover (classic trend-following). Bullish when 50>200.
    ma50, ma200 = p.rolling(50).mean(), p.rolling(200).mean()
    sig['trend_ma_50_200'] = np.sign(ma50 - ma200)

    # 3. PRICE vs 200d MA (simple trend filter), continuous.
    sig['px_vs_ma200'] = _z(p / ma200 - 1)

    # 4. TSMOM time-series momentum, 63d (3mo) and 126d (6mo) past return.
    sig['tsmom_63'] = np.sign(p / p.shift(63) - 1)
    sig['tsmom_126'] = np.sign(p / p.shift(126) - 1)

    # 5. SHORT-TERM REVERSAL (mean reversion): 5d move, contrarian → bullish after dump.
    sig['reversal_5d'] = -_z(p / p.shift(5) - 1, win=126)

    # 6. RSI(14) mean-reversion: oversold (<30) bullish, overbought (>70) bearish.
    sig['rsi_14'] = -(_rsi(p) - 50) / 50.0       # oversold -> positive(bullish)

    # 7. STORAGE surprise (fundamental): high storage = oversupply = bearish.
    if 'eia_storage_weekly' in df.columns:
        st = df['eia_storage_weekly'].astype(float).ffill()
        sig['storage_z'] = -_z(st)               # high storage -> bearish

    # 8. CROSS-ASSET: crude (CL) momentum as an energy-complex driver.
    if 'CL' in df.columns:
        cl = df['CL'].astype(float)
        sig['crude_mom_21'] = np.sign(cl / cl.shift(21) - 1)

    # 9. SEASONALITY: NG strong Nov-Feb (winter demand), weak Mar-May/Sep-Oct.
    mon = df.index.month
    seas = pd.Series(0.0, index=df.index)
    seas[np.isin(mon, [11, 12, 1, 2])] = 1.0     # winter -> bullish
    seas[np.isin(mon, [3, 4, 5, 9, 10])] = -1.0  # shoulder -> bearish
    sig['seasonality'] = seas

    # 10. REALIZED-VOL momentum (vol-of-vol regime): rising RV often precedes drops.
    rv = p.pct_change().rolling(20).std() * np.sqrt(252)
    sig['rising_vol'] = -_z(rv.diff(5))          # rising vol -> bearish

    # 11. IV-RANK factor: top-quintile real ATM IV → -23% fwd-63d (p=.002, our
    #     prior finding). High IV-rank → bearish-direction signal.
    ivp = os.path.join(CACHE, 'ung_iv_rank_daily.csv')
    if os.path.exists(ivp):
        ivr = pd.read_csv(ivp, index_col=0, parse_dates=True)['iv_rank']
        sig['iv_rank'] = -(ivr.reindex(df.index, method='ffill', limit=10) - 0.5) * 2

    # 12-15. WEATHER / DEMAND: degree-day anomaly vs seasonal normal. More heating
    #     (HDD) or cooling (CDD) demand than normal = more gas burn = BULLISH.
    #     Seasonal normal = day-of-year climatology (mild in-sample lookahead in the
    #     normal only; removes the deterministic seasonal cycle, not the anomaly).
    ddp = os.path.join(CACHE, 'degree_days_daily.csv')
    if os.path.exists(ddp):
        dd = pd.read_csv(ddp, index_col=0, parse_dates=True)
        doy = dd.index.dayofyear
        hdd_a = dd['hdd'] - dd['hdd'].groupby(doy).transform('mean')
        cdd_a = dd['cdd'] - dd['cdd'].groupby(doy).transform('mean')
        tot_a = hdd_a + cdd_a
        rdx = lambda s: s.reindex(df.index, method='ffill', limit=3)
        sig['demand_anom'] = _z(rdx(tot_a))                 # total degree-day anomaly
        sig['demand_anom_7d'] = _z(rdx(tot_a.rolling(7).mean()))  # 7d-smoothed
        sig['hdd_anom'] = _z(rdx(hdd_a))                    # heating only
        sig['cdd_anom'] = _z(rdx(cdd_a))                    # cooling only

    return sig


def fwd_returns(df, horizons=(5, 21, 63)):
    p = df['UNG'].astype(float)
    out = {}
    for h in horizons:
        out[h] = p.shift(-h) / p - 1
    return out


def block_bootstrap_ic_p(sig, ret, block=21, n=1000, seed=42):
    """Two-sided p-value that IC != 0, via circular block bootstrap (handles
    overlapping-return autocorrelation)."""
    d = pd.concat([sig, ret], axis=1).dropna()
    if len(d) < block * 4:
        return np.nan
    s, r = d.iloc[:, 0].values, d.iloc[:, 1].values
    obs = np.corrcoef(pd.Series(s).rank(), pd.Series(r).rank())[0, 1]
    m = len(s)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        idx = (np.concatenate([np.arange(st, st + block)
               for st in rng.integers(0, m, size=m // block + 1)])[:m]) % m
        rb = r[idx]  # shuffle returns in blocks vs fixed signal -> null
        ic = np.corrcoef(pd.Series(s).rank(), pd.Series(rb).rank())[0, 1]
        if abs(ic) >= abs(obs):
            cnt += 1
    return cnt / n


def long_flat_backtest(sig, p):
    """Daily-rebalanced long-when-bullish / flat-when-not. Non-overlapping P&L."""
    pos = (sig > 0).astype(float).shift(1).fillna(0)  # act next day, no lookahead
    r = p.pct_change().fillna(0)
    strat = pos * r
    d = strat.dropna()
    if len(d) < 60 or d.std() == 0:
        return None
    ann = d.mean() * 252
    sh = d.mean() / d.std() * np.sqrt(252)
    eq = (1 + strat).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    exposure = pos.mean()
    return {'ann': ann, 'sharpe': sh, 'mdd': mdd, 'exposure': exposure}


def main():
    df = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = df[df['UNG'].notna()].copy()
    sig = build_signals(df)
    fwd = fwd_returns(df)
    p = df['UNG'].astype(float)

    # buy-and-hold benchmark
    r = p.pct_change().dropna()
    bh = {'ann': r.mean() * 252, 'sharpe': r.mean() / r.std() * np.sqrt(252)}
    print(f"UNG buy-and-hold: ann {bh['ann']:+.1%}  Sharpe {bh['sharpe']:+.2f}  "
          f"({df.index[0].date()}→{df.index[-1].date()})\n")

    rows = []
    for name in sig.columns:
        s = sig[name]
        ic = {}
        for h, fr in fwd.items():
            d = pd.concat([s, fr], axis=1).dropna()
            ic[h] = (d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())
                     if len(d) > 60 else np.nan)
        pval21 = block_bootstrap_ic_p(s, fwd[21])
        bt = long_flat_backtest(s, p)
        rows.append({
            'signal': name,
            'IC_5d': ic[5], 'IC_21d': ic[21], 'IC_63d': ic[63],
            'p(IC21)': pval21,
            'LF_ann': bt['ann'] if bt else np.nan,
            'LF_sharpe': bt['sharpe'] if bt else np.nan,
            'LF_mdd': bt['mdd'] if bt else np.nan,
            'LF_expo': bt['exposure'] if bt else np.nan,
        })
    res = pd.DataFrame(rows).set_index('signal')
    res = res.reindex(res['IC_21d'].abs().sort_values(ascending=False).index)

    pd.set_option('display.width', 160, 'display.float_format', lambda x: f'{x:+.3f}')
    print("DIRECTIONAL EDGE (sorted by |IC_21d|).  IC = Spearman(signal, fwd return).")
    print("  IC>0 = signal predicts direction. |IC|>~0.05 w/ p<0.1 = worth a look.")
    print("  LF_* = long-when-bullish/flat daily backtest (vs B&H above).\n")
    print(res.to_string())
    print("\nReading: a real timing edge needs IC same-sign across horizons, p<0.10,")
    print("AND LF_sharpe > buy-and-hold at < full exposure. Most won't clear it.")
    res.to_csv(os.path.join(THIS, 'results', 'timing_signals_eval.csv'))

    # EVENT STUDY: extreme backwardation (the original claim was about RARE >$0.40
    # events, which a continuous IC washes out). Does the tail actually predict?
    if 'eia_hh_spot_daily' in df.columns:
        basis = (df['eia_hh_spot_daily'].astype(float) - df['NG'].astype(float)).ffill(limit=5)
        print("\n--- EVENT STUDY: extreme backwardation (HH spot − futures) ---")
        for q, lbl in [(0.90, 'top-decile'), (0.95, 'top-5%')]:
            thr = basis.quantile(q)
            mask = basis >= thr
            for h in (5, 21):
                cond = fwd[h][mask].dropna()
                base = fwd[h].dropna()
                print(f"  basis≥{thr:+.2f} ({lbl}, n={mask.sum()}): fwd{h}d "
                      f"mean {cond.mean():+.1%} vs unconditional {base.mean():+.1%} "
                      f"(hit-rate down {100*(cond<0).mean():.0f}%)")


if __name__ == '__main__':
    main()
