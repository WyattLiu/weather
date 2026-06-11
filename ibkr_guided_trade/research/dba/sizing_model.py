"""Fundamental expected-return model → Kelly-style DBA wheel sizing.

Model: walk-forward ridge regression, refit monthly on expanding window
(min 36 months), monthly-sampled to avoid overlap inflation:

    E[r_63d] = ridge(oni, cot_chg_13w, stu_z, fpi_mom_3m,
                     dxy_trend, crude_3m, month_sin, month_cos)

Sizing (fractional-Kelly, upsize-only per
[[feedback_filters_cost_more_than_they_save]]):

    kelly_raw  = E[r] / sigma63^2          (optimal fraction)
    size_mult  = clip(1 + 0.5 * kelly_raw, 1.0, 2.2)   on E[r] > 0
    size_mult  = 1.0                                   on E[r] <= 0

MACRO-SQUEEZE CAP (from drawdown_forensics): when >=2 of
{dxy_rising, crude_rising, cot_flow_hot} fire (lifts 3.3x/2.7x/2.3x at
drawdown peaks), cap size_mult at 1.0 — never below (upsize-only law),
but don't lever into the regime that precedes every modern decline.

Validation: wheel backtest with model-driven signal_fn vs threshold
combo vs baseline.
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
sys.path.insert(0, THIS_DIR)

# Regression features need LONG history (COT 2010+, others 1990s+).
# dxy_trend/crude_3m only exist 2021+ (master_dataset window) — they'd
# delay training to 2024 and starve validation, so they serve ONLY in
# the macro-squeeze warning cap (forensics lift 3.3x/2.7x), not in E[r].
FEATURES = ['oni', 'cot_chg_13w', 'stu_z', 'fpi_mom_3m',
            'month_sin', 'month_cos']


def build_panel():
    facs = pd.read_csv(os.path.join(CACHE, 'dba_factor_panel.csv'),
                       index_col=0, parse_dates=True)
    fund = pd.read_csv(os.path.join(CACHE, 'dba_fundamentals_panel.csv'),
                       index_col=0, parse_dates=True)
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    dba = panel['DBA'].dropna()

    f = pd.DataFrame(index=dba.index)
    f['oni'] = facs['oni']
    f['dxy_trend'] = facs['dxy_trend']
    f['cot_chg_13w'] = fund['cot_chg_13w']
    f['stu_z'] = fund['stu_z']
    f['fpi_mom_3m'] = fund['fpi_mom_3m']
    f['crude_3m'] = fund['crude_3m']
    f['month_sin'] = np.sin(2 * np.pi * f.index.month / 12)
    f['month_cos'] = np.cos(2 * np.pi * f.index.month / 12)
    f['sigma63'] = dba.pct_change().rolling(63).std() * math.sqrt(63)  # 63d-horizon vol
    f['fwd_63d'] = dba.pct_change(63).shift(-63)
    # lag observables 1d (warning fields too, even though not in FEATURES)
    for c in FEATURES + ['sigma63', 'dxy_trend', 'crude_3m']:
        f[c] = f[c].shift(1)
    return f


def ridge_fit(X, y, lam=1.0):
    """Standardized ridge with intercept. Returns predict(x_row)."""
    mu, sd = X.mean(0), X.std(0).replace(0, 1)
    Xs = (X - mu) / sd
    Xb = np.column_stack([np.ones(len(Xs)), Xs.values])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam  # don't penalize intercept
    w = np.linalg.solve(A, Xb.T @ y.values)

    def predict(row):
        xs = ((row - mu) / sd).values
        return float(w[0] + xs @ w[1:])
    return predict, w


def walk_forward(f, min_train_months=36):
    """Monthly refit; predict E[r_63d] for each day of the next month."""
    monthly = f.dropna(subset=FEATURES + ['fwd_63d']).iloc[::21]
    days = f.dropna(subset=FEATURES)
    preds = pd.Series(index=days.index, dtype=float)
    coefs = []
    month_starts = pd.date_range(monthly.index[0], days.index[-1], freq='MS')
    for i, ms in enumerate(month_starts):
        train = monthly[monthly.index < ms - pd.Timedelta(days=70)]  # embargo fwd overlap
        if len(train) < min_train_months:
            continue
        predict, w = ridge_fit(train[FEATURES], train['fwd_63d'])
        nxt = days[(days.index >= ms)
                   & (days.index < ms + pd.offsets.MonthBegin(1))]
        for d, row in nxt[FEATURES].iterrows():
            preds.loc[d] = predict(row)
        coefs.append({'month': str(ms.date()),
                      **{c: round(float(v), 4)
                         for c, v in zip(['int'] + FEATURES, w)}})
    return preds.dropna(), pd.DataFrame(coefs)


def size_from_pred(er, sigma63, warn_count, kelly_frac=0.5,
                   cap=2.2):
    if pd.isna(er) or pd.isna(sigma63) or sigma63 <= 0:
        return 1.0
    if warn_count >= 2:
        return 1.0  # macro-squeeze cap
    if er <= 0:
        return 1.0  # upsize-only
    kelly = er / (sigma63 ** 2)
    return float(np.clip(1 + kelly_frac * kelly * 0.1, 1.0, cap))


def main():
    f = build_panel()
    print('[model] walk-forward ridge, monthly refits...')
    preds, coefs = walk_forward(f)
    print(f'  {len(preds)} daily predictions '
          f'{preds.index[0].date()} → {preds.index[-1].date()}')

    # Out-of-sample IC
    join = pd.DataFrame({'pred': preds, 'real': f['fwd_63d']}).dropna().iloc[::21]
    ic = join['pred'].corr(join['real'])
    ic_rank = join['pred'].corr(join['real'], method='spearman')
    hit = ((join['pred'] > 0) == (join['real'] > 0)).mean()
    print(f'\n=== OOS validity (monthly samples, n={len(join)}) ===')
    print(f'  IC (pearson):  {ic:+.3f}')
    print(f'  IC (spearman): {ic_rank:+.3f}')
    print(f'  direction hit: {hit:.1%}')

    # latest coefficients (interpretation)
    print('\n=== Latest model coefficients (standardized) ===')
    last = coefs.iloc[-1]
    for c in FEATURES:
        print(f'  {c:<14} {last[c]:+.4f}')

    # warnings (macro-squeeze components)
    warns = ((f['dxy_trend'] > 0.02).astype(int)
             + (f['crude_3m'] > 0.10).astype(int)
             + (f['cot_chg_13w'] > 0.05).astype(int))

    # sizing series
    size = pd.Series(
        {d: size_from_pred(preds.get(d), f.loc[d, 'sigma63'], warns.get(d, 0))
         for d in preds.index})
    print(f'\nsize_mult: mean {size.mean():.2f}, >1 on {(size > 1).mean():.0%} '
          f'of days, capped-by-warning on {(warns.loc[size.index] >= 2).mean():.0%}')

    # save model state for live use
    out = {
        'as_of': str(preds.index[-1].date()),
        'er_63d': round(float(preds.iloc[-1]), 4),
        'sigma63': round(float(f['sigma63'].iloc[-1]), 4),
        'warn_count': int(warns.iloc[-1]),
        'size_mult': round(float(size.iloc[-1]), 2),
        'oos_ic': round(float(ic), 3),
        'oos_hit': round(float(hit), 3),
        'coefficients': coefs.iloc[-1].to_dict(),
    }
    with open(os.path.join(CACHE, 'sizing_model_state.json'), 'w') as fjs:
        json.dump(out, fjs, indent=2)
    size.to_csv(os.path.join(CACHE, 'sizing_model_series.csv'))
    coefs.to_csv(os.path.join(CACHE, 'sizing_model_coefs.csv'), index=False)
    print(f"\nLive state: E[r63]={out['er_63d']:+.2%}  warn={out['warn_count']}  "
          f"size_mult={out['size_mult']}")
    print(f'→ {CACHE}/sizing_model_state.json')


if __name__ == '__main__':
    main()
