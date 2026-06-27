"""Trade-level forensics — the repeatable post-tournament study.

Run after any replay tournament to dissect a kernel's behavior trade by
trade and day by day. Produces the standard six-section report that
drove the gen-4 design (see KERNEL_LAB.md):

  1. P&L by trade type (win rates, averages) + cycle nets (put vs call)
  2. Worst-10 NAV days with spot context (leverage of the loss)
  3. Roll futility (rolls followed by assignment anyway)
  4. Loss clustering by month
  5. Drawdown episodes >5% (start/end/depth/length)
  6. Cascade days (many defensive trades on one date)

Usage:
    venv/bin/python backtest/trade_forensics.py --strategy champion_smooth_ddtrim_ivrank
    venv/bin/python backtest/trade_forensics.py --strategy g3_full_stack
"""
import os
import argparse
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(THIS_DIR, 'results')


def main(strategy):
    t = pd.read_csv(os.path.join(RESULTS, f'{strategy}_trades.csv'),
                    parse_dates=['date'])
    h = pd.read_csv(os.path.join(RESULTS, f'{strategy}_history.csv'))
    h['date'] = pd.to_datetime(h['date']).dt.normalize()
    h = h.set_index('date')
    h['ret'] = h['nav'].pct_change()

    print(f'=== FORENSICS: {strategy} ({len(t)} trades, {len(h)} days) ===')

    # 1. P&L by type + cycle nets
    pnl = t[t['pnl'].notna() & (t['pnl'] != 0)]
    g = pnl.groupby('type')['pnl'].agg(
        total='sum', n='count', avg='mean', win_rate=lambda x: (x > 0).mean())
    print('\n-- 1. P&L by trade type --')
    print(g.sort_values('total').round(1).to_string())
    put_types = [c for c in g.index if 'PUT' in c and 'LONG' not in c]
    call_types = [c for c in g.index if 'CALL' in c or 'CC' in c or 'ELEVATOR' in c]
    print(f"\n   PUT cycle net:  ${g.loc[[c for c in put_types if c in g.index], 'total'].sum():>12,.0f}")
    print(f"   CALL cycle net: ${g.loc[[c for c in call_types if c in g.index], 'total'].sum():>12,.0f}")

    # 2. Worst days
    print('\n-- 2. Worst 10 NAV days --')
    spot_ret = h['spot'].pct_change()
    for d, r in h.nsmallest(10, 'ret').iterrows():
        sr = spot_ret.get(d, 0)
        lev = r['ret'] / sr if sr else float('nan')
        print(f"  {d.date()}  nav {r['ret']:+.1%}  spot {sr:+.1%}  (x{lev:.1f})")

    # 3. Roll futility
    for roll_type, assign_type in (('PUT_ROLL_DOWN', 'PUT_ASSIGN'),
                                   ('CALL_ROLL_UP', 'CALL_ASSIGN')):
        rolls = t[t['type'] == roll_type]
        assigns = t[t['type'] == assign_type]
        futile = sum(
            1 for _, r in rolls.iterrows()
            if len(assigns[(assigns['date'] > r['date'])
                           & (assigns['date'] <= r['date'] + pd.Timedelta(days=60))]))
        if len(rolls):
            print(f'\n-- 3. {roll_type}: {len(rolls)} events, '
                  f'${rolls["pnl"].sum():,.0f}; {futile} ({futile/len(rolls):.0%}) '
                  f'followed by {assign_type} within 60d')

    # 4. Loss months
    losses = t[t['pnl'] < -100].copy()
    losses['ym'] = losses['date'].dt.to_period('M')
    print('\n-- 4. Worst loss months --')
    print(losses.groupby('ym')['pnl'].sum().nsmallest(6).round(0).to_string())

    # 5. DD episodes
    nav = h['nav']
    dd = nav / nav.cummax() - 1
    print('\n-- 5. Drawdown episodes >5% --')
    start = None
    for d, v in (dd < -0.05).items():
        if v and start is None:
            start = d
        elif not v and start is not None:
            print(f'  {start.date()} → {d.date()}  depth {dd.loc[start:d].min():.1%}  '
                  f'({(d-start).days}d)')
            start = None
    if start is not None:
        print(f'  {start.date()} → open  depth {dd.loc[start:].min():.1%}')

    # 6. Cascade days (defensive-trade pileups)
    defensive = t[t['type'].isin(['PUT_ROLL_DOWN', 'CALL_ROLL_UP', 'DD_TRIM_SHARES'])]
    casc = defensive.groupby(defensive['date'].dt.date).size().nlargest(6)
    print('\n-- 6. Cascade days (defensive trades per day) --')
    print(casc.to_string())

    # 7. INTEGRITY / BUG SCREEN — every review must hunt for the bug
    # families that have actually bitten us (see KERNEL_LAB.md):
    # CC-stacking (naked calls), leverage cascades (negative cash),
    # marking noise (NAV swings >> spot), collateral violations.
    print('\n-- 7. INTEGRITY / BUG SCREEN --')
    flags = 0
    if {'shares', 'short_calls'}.issubset(h.columns):
        naked = h[h['short_calls'] * 100 > h['shares'] + 1]
        if len(naked):
            flags += 1
            print(f'  ⚠ COVERED-CALL VIOLATION: {len(naked)} days with short_calls*100 > shares '
                  f'(first: {naked.index[0].date()}, worst: '
                  f'{int((naked["short_calls"]*100 - naked["shares"]).max())} naked-share-equiv)')
        else:
            print('  ✓ covered-calls-only holds on every day')
    if 'cash' in h.columns:
        neg = h[h['cash'] < -1000]
        if len(neg):
            flags += 1
            print(f'  ⚠ NEGATIVE CASH (leverage): {len(neg)} days, '
                  f'min ${h["cash"].min():,.0f} on {h["cash"].idxmin().date()}')
        else:
            print('  ✓ cash never goes negative (no hidden leverage)')
    if {'shares', 'short_puts', 'cash'}.issubset(h.columns):
        # cash-secured proxy: put collateral vs cash+shares value
        approx_collat = h['short_puts'] * h['spot'] * 100 * 0.95
        breach = h[approx_collat > (h['cash'] + h['shares'] * h['spot']) * 1.05]
        if len(breach):
            flags += 1
            print(f'  ⚠ COLLATERAL STRETCH: {len(breach)} days where est. put collateral '
                  f'> account value (margin dependence)')
        else:
            print('  ✓ put collateral within account value all days')
    # marking-noise screen: NAV move >> spot move with no position change
    spot_ret = h['spot'].pct_change()
    sus = h[(h['ret'].abs() > 0.08) & (spot_ret.abs() < 0.02)]
    if len(sus):
        flags += 1
        print(f'  ⚠ MARKING-NOISE SUSPECTS: {len(sus)} days with |NAV ret|>8% on |spot|<2% '
              f'(e.g. {sus.index[0].date()}) — check liability marks')
    else:
        print('  ✓ no NAV-vs-spot dislocations >8%/2%')
    # by-construction win rates (0%/100%) are attribution artifacts, not skill
    arti = g[(g['win_rate'].isin([0.0, 1.0])) & (g['n'] > 20)]
    if len(arti):
        print(f'  ℹ {len(arti)} trade types have 0%/100% win rates BY CONSTRUCTION '
              f'(cost/income legs) — never read them as skill')
    # stale data screen
    stale = (h['spot'].diff() == 0).rolling(5).sum()
    if (stale >= 5).any():
        flags += 1
        print('  ⚠ STALE SPOT: 5+ consecutive unchanged closes detected — check data feed')
    else:
        print('  ✓ no stale-price runs')
    print(f'  → integrity flags: {flags}')

    # walk-forward floor for the header table
    w12 = nav.pct_change(252).dropna()
    if len(w12):
        print(f'\nworst-12mo: {w12.min():+.1%}   best-12mo: {w12.max():+.1%}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', required=True)
    main(ap.parse_args().strategy)
