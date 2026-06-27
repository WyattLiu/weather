"""DBA drawdown forensics — what caused the big declines, and what
factor states preceded them.

For every drawdown >10% from peak (2007-2026):
  - peak/trough dates, depth, length, recovery
  - factor state AT THE PEAK (1 week before): stocks-to-use z, COT flow,
    DXY trend, ONI, FPI momentum, crude trend
  - which factors were in their "warning" quintile

Output: episode table + warning-factor hit rates → tells us which
fundamentals LEAD declines (sizing-down candidates are banned per
upsize-only law, but warnings can cap the upsize at 1.0x).
"""
import os
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')


def find_drawdowns(spot, min_depth=0.10, window=252):
    """Episodes vs ROLLING 1y peak — DBA never regained its 2008 high
    (structural futures-roll decay), so all-time-peak drawdowns collapse
    into one 12-year episode. Rolling peaks match the wheel's horizon."""
    roll_max = spot.rolling(window, min_periods=20).max()
    dd = spot / roll_max - 1
    episodes = []
    in_ep = False
    peak_date = None
    trough_date, trough_val = None, 0.0
    for d, v in dd.items():
        if not in_ep and v <= -min_depth:
            in_ep = True
            # backtrack: the peak is the date of the rolling max
            win = spot.loc[:d].tail(window)
            peak_date = win.idxmax()
            trough_date, trough_val = d, v
        elif in_ep:
            if v < trough_val:
                trough_val, trough_date = v, d
            if v >= -0.005:  # recovered to the rolling high
                episodes.append({'peak': peak_date, 'trough': trough_date,
                                 'recovered': d, 'depth': round(trough_val, 4),
                                 'days_down': (trough_date - peak_date).days})
                in_ep = False
    if in_ep:
        episodes.append({'peak': peak_date, 'trough': trough_date,
                         'recovered': None, 'depth': round(trough_val, 4),
                         'days_down': (trough_date - peak_date).days})
    return episodes


def main():
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    dba = panel['DBA'].dropna()

    facs = pd.read_csv(os.path.join(CACHE, 'dba_factor_panel.csv'),
                       index_col=0, parse_dates=True)
    fund = pd.read_csv(os.path.join(CACHE, 'dba_fundamentals_panel.csv'),
                       index_col=0, parse_dates=True)
    F = facs[['oni', 'dxy_trend', 'ng_trend']].join(
        fund[['cot_chg_13w', 'fpi_mom_3m', 'stu_z', 'crude_3m']], how='outer')

    # "warning" thresholds from the scan signs (bad-for-DBA直):
    # high oni, high stu_z, high cot_chg (hot flow), high dxy/crude/ng trend,
    # NEGATIVE fpi momentum
    warn_rules = {           # (field, predicate)
        'oni_high': ('oni', lambda v: v > 0.5),
        'stocks_loose': ('stu_z', lambda v: v > 0.5),
        'cot_flow_hot': ('cot_chg_13w', lambda v: v > 0.05),
        'dxy_rising': ('dxy_trend', lambda v: v > 0.02),
        'crude_rising': ('crude_3m', lambda v: v > 0.10),
        'fpi_falling': ('fpi_mom_3m', lambda v: v < -0.01),
    }

    episodes = find_drawdowns(dba, min_depth=0.10)
    rows = []
    for ep in episodes:
        # factor state one week before the peak
        ref = ep['peak'] - pd.Timedelta(days=7)
        state = F.loc[:ref].iloc[-1] if len(F.loc[:ref]) else pd.Series(dtype=float)
        flags = {}
        for k, (field, pred) in warn_rules.items():
            v = state.get(field)
            flags[k] = None if (v is None or pd.isna(v)) else bool(pred(v))
        n_warn = sum(1 for v in flags.values() if v is True)
        rows.append({
            'peak': str(ep['peak'].date()), 'trough': str(ep['trough'].date()),
            'depth': f"{ep['depth']:.0%}", 'days_down': ep['days_down'],
            'recovered': str(ep['recovered'].date()) if ep['recovered'] is not None else 'OPEN',
            'n_warnings': n_warn,
            **{k: ('Y' if v is True else '-' if v is False else '?')
               for k, v in flags.items()},
        })
    # n_warnings as share of KNOWN flags
    for row in rows:
        known = [c for c in warn_rules if row[c] != '?']
        row['warn_share'] = (f"{sum(row[c] == 'Y' for c in known)}/{len(known)}"
                             if known else 'n/a')
    df = pd.DataFrame(rows)
    print('=== DBA DRAWDOWNS >10% (2007-2026) — factor state at peak ===\n')
    print(df.to_string(index=False))

    # hit rates at peaks vs base rates (lift = leading-signal value)
    print('\n=== Warning hit rate at drawdown peaks vs base rate (lift) ===')
    for k, (field, pred) in warn_rules.items():
        col = df[k]
        known = col[col != '?']
        base = F[field].dropna().apply(pred).mean() if field in F.columns else float('nan')
        if len(known):
            hit = (known == 'Y').mean()
            lift = hit / base if base and base > 0 else float('nan')
            print(f'  {k:<15} peak={hit:.0%}  base={base:.0%}  lift={lift:.1f}x  ({len(known)} eps)')

    df.to_csv(os.path.join(CACHE, 'dba_drawdown_forensics.csv'), index=False)
    print(f'\n→ {CACHE}/dba_drawdown_forensics.csv')


if __name__ == '__main__':
    main()
