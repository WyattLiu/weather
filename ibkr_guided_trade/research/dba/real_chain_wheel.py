"""Wheel backtest against REAL historical option chains (ThetaData).

The final fidelity rung: no BSM, no synthetic strikes, no synthetic
expiries. Uses actual EOD quotes from research/gex/history/thetadata/:
  - entries SELL at the actual BID (conservative)
  - take-profit BUYS at the actual ASK when ask <= 50% of entry credit
  - strikes: only those actually quoted that day
  - expiries: only the monthlies that actually existed
  - daily liability marked at mid (fallback last close mark)

Run: venv/bin/python research/dba/real_chain_wheel.py --symbol DBA
"""
import os
import sys
import glob
import math
import argparse
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
TD = os.path.join(ROOT, 'research', 'gex', 'history', 'thetadata')
CACHE = os.path.join(THIS_DIR, 'cache')


def load_chain_panel(symbol):
    """All EOD quotes → dict[(quote_date, expiry, right)] = DataFrame(strike rows)."""
    parts = []
    for p in sorted(glob.glob(os.path.join(TD, symbol.lower(), '*_eod.csv'))):
        try:
            df = pd.read_csv(p)
            if len(df):
                parts.append(df)
        except Exception:
            continue
    eod = pd.concat(parts, ignore_index=True)
    eod['quote_date'] = pd.to_datetime(eod['quote_date'])
    eod['expiry'] = pd.to_datetime(eod['expiry'])
    return eod


def run_real_wheel(symbol, otm_pct=0.02, dte_target=60, dte_lo=30, dte_hi=80,
                   tp_pct=0.50, entry_cadence_days=7,
                   contract_per_nav=15000, init_nav=100000):
    eod = load_chain_panel(symbol)
    spot = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                       index_col=0, parse_dates=True)[symbol].dropna()
    # quotes index for fast lookup
    eod = eod.set_index(['quote_date', 'expiry', 'right']).sort_index()

    dates = sorted(set(eod.index.get_level_values(0)) & set(spot.index))
    cash, shares, nav = init_nav, 0, init_nav
    open_puts, open_calls = [], []   # dicts: expiry,K,entry_credit,n,last_mark
    last_entry = None
    curve, trades = [], []

    def quote(d, exp, right, K):
        try:
            sub = eod.loc[(d, exp, right)]
            row = sub[sub['strike'] == K]
            if len(row):
                r = row.iloc[0]
                bid, ask = float(r['bid']), float(r['ask'])
                # thin chains print garbage asks (0.05/5.00) — mark at BID
                # (floored at intrinsic later); mid only when spread is sane
                if bid > 0 and ask > 0 and ask <= bid * 3 + 0.10:
                    mark = (bid + ask) / 2
                elif bid > 0:
                    mark = bid
                else:
                    mark = float(r['close'])
                return bid, ask, mark
        except KeyError:
            pass
        return None, None, None

    for d in dates:
        S = float(spot.loc[d])

        # mark/manage open puts
        keep = []
        for pos in open_puts:
            bid, ask, mid = quote(d, pos['expiry'], 'P', pos['K'])
            if mid is not None:
                pos['last_mark'] = mid
            if d >= pos['expiry']:
                if S < pos['K']:
                    shares += 100 * pos['n']
                    cash -= pos['K'] * 100 * pos['n']
                    trades.append({'date': d, 'type': 'assign', 'K': pos['K']})
                else:
                    trades.append({'date': d, 'type': 'put_otm', 'K': pos['K']})
                continue
            if ask is not None and ask > 0 and ask <= pos['entry_credit'] * (1 - tp_pct):
                cash -= ask * 100 * pos['n']
                trades.append({'date': d, 'type': 'put_tp', 'K': pos['K'],
                               'kept': (pos['entry_credit'] - ask) * 100 * pos['n']})
                continue
            keep.append(pos)
        open_puts = keep

        keep = []
        for pos in open_calls:
            bid, ask, mid = quote(d, pos['expiry'], 'C', pos['K'])
            if mid is not None:
                pos['last_mark'] = mid
            if d >= pos['expiry']:
                if S > pos['K'] and shares >= 100 * pos['n']:
                    shares -= 100 * pos['n']
                    cash += pos['K'] * 100 * pos['n']
                    trades.append({'date': d, 'type': 'called_away', 'K': pos['K']})
                continue
            if ask is not None and ask > 0 and ask <= pos['entry_credit'] * (1 - tp_pct):
                cash -= ask * 100 * pos['n']
                trades.append({'date': d, 'type': 'cc_tp', 'K': pos['K']})
                continue
            keep.append(pos)
        open_calls = keep

        # entries
        if last_entry is None or (d - last_entry).days >= entry_cadence_days:
            # available expiries with quotes today
            try:
                today_slice = eod.loc[d]
                exps = sorted({e for e, r in today_slice.index
                               if dte_lo <= (e - d).days <= dte_hi})
            except KeyError:
                exps = []
            if exps:
                exp = min(exps, key=lambda e: abs((e - d).days - dte_target))
                # real strikes for puts that day
                try:
                    pk = eod.loc[(d, exp, 'P')]
                    strikes = pk[pk['bid'] > 0]['strike'].values
                except KeyError:
                    strikes = []
                if len(strikes):
                    target = S * (1 - otm_pct)
                    K = min(strikes, key=lambda k: abs(k - target))
                    bid, ask, mid = quote(d, exp, 'P', K)
                    if bid and bid > 0.05:
                        n = max(1, min(25, int(nav / contract_per_nav)))
                        # CASH-SECURED: cap by cash NET of collateral already
                        # committed to open puts (prevents assignment-cascade
                        # leverage that produced 477-lot positions)
                        committed = sum(p['K'] * 100 * p['n'] for p in open_puts)
                        free_cash = cash - committed - 5000
                        n = min(n, int(max(0, free_cash) / (K * 100)))
                        if n > 0:
                            cash += bid * 100 * n   # SELL AT BID
                            open_puts.append({'expiry': exp, 'K': K,
                                              'entry_credit': bid, 'n': n,
                                              'last_mark': mid or bid})
                            trades.append({'date': d, 'type': 'put_open',
                                           'K': K, 'credit': bid, 'n': n})
                # CC if assigned shares
                if shares >= 100:
                    try:
                        ck = eod.loc[(d, exp, 'C')]
                        cstrikes = ck[ck['bid'] > 0]['strike'].values
                    except KeyError:
                        cstrikes = []
                    if len(cstrikes):
                        Kc = min(cstrikes, key=lambda k: abs(k - S * (1 + otm_pct)))
                        bid, ask, mid = quote(d, exp, 'C', Kc)
                        if bid and bid > 0.05:
                            ncc = shares // 100
                            cash += bid * 100 * ncc
                            open_calls.append({'expiry': exp, 'K': Kc,
                                               'entry_credit': bid, 'n': ncc,
                                               'last_mark': mid or bid})
                            trades.append({'date': d, 'type': 'cc_open', 'K': Kc})
                last_entry = d

        # INTRINSIC-ONLY liability marking. Thin-chain quotes (2017-2019
        # especially) are too noisy to mark against — wide/stale prints
        # caused fake ±5x NAV swings. All CASHFLOWS remain real (entry at
        # bid, TP at ask, assignment at strike); NAV just amortizes the
        # extrinsic over the position's life instead of mark-to-noise.
        liability = sum(max(p['K'] - S, 0) * 100 * p['n'] for p in open_puts)
        liability += sum(max(S - c['K'], 0) * 100 * c['n'] for c in open_calls)
        nav = cash + shares * S - liability
        curve.append({'date': d, 'nav': nav, 'spot': S, 'shares': shares})

    return pd.DataFrame(curve).set_index('date'), pd.DataFrame(trades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='DBA')
    args = ap.parse_args()
    curve, trades = run_real_wheel(args.symbol)
    ret = curve['nav'].pct_change().dropna()
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    ann = (curve['nav'].iloc[-1] / curve['nav'].iloc[0]) ** (1 / yrs) - 1
    sharpe = ret.mean() / ret.std() * math.sqrt(252)
    mdd = (curve['nav'] / curve['nav'].cummax() - 1).min()
    print(f'{args.symbol} REAL-CHAIN wheel ({curve.index[0].date()} → '
          f'{curve.index[-1].date()}, {yrs:.1f}y): '
          f'ann={ann:+.2%}  sharpe={sharpe:+.2f}  mdd={mdd:.2%}  '
          f'trades={len(trades)}')
    curve.to_csv(os.path.join(CACHE, f'real_wheel_{args.symbol.lower()}_curve.csv'))


if __name__ == '__main__':
    main()
