#!/usr/bin/env python3
"""
Scan NG (Natural Gas) futures call options on NYMEX via IBKR.

Usage:
    python scan_ng_options.py                  # Scan calls, 10-120 DTE
    python scan_ng_options.py --right P        # Scan puts
    python scan_ng_options.py --min-dte 30 --max-dte 180
    python scan_ng_options.py --all-months     # Scan options on multiple futures months
"""

import argparse
from ib_insync import IB, Future, FuturesOption
from datetime import datetime
from modules.common import IBKR_HOST, IBKR_PORT


def get_ng_futures(ib, max_months=12):
    """Get NG futures sorted by expiration."""
    ng = Future('NG', exchange='NYMEX')
    contracts = ib.reqContractDetails(ng)

    futs = []
    for cd in contracts:
        c = cd.contract
        exp = c.lastTradeDateOrContractMonth
        exp_dt = datetime.strptime(exp, '%Y%m%d') if len(exp) == 8 else datetime.strptime(exp + '01', '%Y%m%d')
        dte = (exp_dt - datetime.now()).days
        if 0 < dte < 365:
            futs.append((c, exp, dte))

    futs.sort(key=lambda x: x[2])
    return futs[:max_months]


def get_spot(ib, contract):
    """Get current price with weekend fallback."""
    ib.qualifyContracts(contract)
    ib.reqMktData(contract)
    ib.sleep(3)
    t = ib.ticker(contract)
    spot = t.last if t.last and t.last > 0 else t.close
    if not spot or not (spot > 0):
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    return spot


def scan_options(ib, fut_contract, spot, right, min_dte, max_dte):
    """Scan options for a specific futures contract."""
    opt_params = ib.reqSecDefOptParams(
        fut_contract.symbol, 'NYMEX', 'FUT', fut_contract.conId)

    if not opt_params:
        opt_params = ib.reqSecDefOptParams(
            fut_contract.symbol, '', 'FUT', fut_contract.conId)

    if not opt_params:
        return []

    # Collect valid expirations and strikes
    all_exps = set()
    all_strikes = set()
    for op in opt_params:
        for exp in op.expirations:
            exp_dt = datetime.strptime(exp, '%Y%m%d')
            dte = (exp_dt - datetime.now()).days
            if min_dte <= dte <= max_dte:
                all_exps.add((exp, dte))
        all_strikes.update(op.strikes)

    # For calls: spot*0.85 to spot*1.60; for puts: spot*0.50 to spot*1.10
    if right == 'C':
        strikes = sorted([s for s in all_strikes if spot * 0.85 <= s <= spot * 1.60])
    else:
        strikes = sorted([s for s in all_strikes if spot * 0.50 <= s <= spot * 1.10])

    # Sort by DTE, skip weeklies if they're too close (< min_dte)
    target_exps = sorted(all_exps, key=lambda x: x[1])

    # Prefer monthly expirations: keep weeklies only if few monthlies exist
    monthlies = [(e, d) for e, d in target_exps
                 if datetime.strptime(e, '%Y%m%d').day >= 20 or d > 20]
    if len(monthlies) >= 3:
        target_exps = monthlies[:8]
    else:
        target_exps = target_exps[:8]

    results = []
    for exp, dte in target_exps:
        opts = [FuturesOption(symbol='NG', lastTradeDateOrContractMonth=exp,
                              strike=s, right=right, exchange='NYMEX')
                for s in strikes]

        try:
            qualified = ib.qualifyContracts(*opts)
        except Exception as e:
            print(f"  Error qualifying {exp}: {e}")
            continue

        valid = [o for o in qualified if o.conId > 0]
        if not valid:
            continue

        for opt in valid:
            ib.reqMktData(opt, '', False, False)
        ib.sleep(4)

        exp_results = []
        for opt in valid:
            t = ib.ticker(opt)
            bid = t.bid if t.bid and t.bid > 0 else 0
            ask = t.ask if t.ask and t.ask > 0 else 0
            if bid > 0 or ask > 0:
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)
                delta = t.modelGreeks.delta if t.modelGreeks else None
                iv = t.modelGreeks.impliedVol if t.modelGreeks else None
                gamma = t.modelGreeks.gamma if t.modelGreeks else None
                theta = t.modelGreeks.theta if t.modelGreeks else None
                otm_pct = ((opt.strike - spot) / spot * 100 if right == 'C'
                           else (spot - opt.strike) / spot * 100)

                r = {
                    'expiry': exp, 'dte': dte, 'strike': opt.strike,
                    'right': right, 'otm_pct': otm_pct,
                    'bid': bid, 'ask': ask, 'mid': mid,
                    'delta': delta, 'iv': iv, 'gamma': gamma, 'theta': theta,
                    'underlying': fut_contract.localSymbol,
                }
                exp_results.append(r)
                results.append(r)

        if exp_results:
            print(f"\n  EXPIRY: {exp} ({dte} DTE) on {fut_contract.localSymbol}")
            print(f"  {'Strike':>8} {'OTM%':>7} {'Bid':>8} {'Ask':>8} {'Mid':>8} "
                  f"{'Delta':>7} {'IV':>7} {'Theta':>7}")
            print(f"  {'-'*66}")
            for r in sorted(exp_results, key=lambda x: x['strike']):
                d_s = f"{r['delta']:.3f}" if r['delta'] else '  N/A'
                iv_s = f"{r['iv']*100:.1f}%" if r['iv'] else '  N/A'
                th_s = f"{r['theta']:.4f}" if r['theta'] else '  N/A'
                print(f"  ${r['strike']:>7.2f} {r['otm_pct']:>+6.1f}% "
                      f"${r['bid']:>7.3f} ${r['ask']:>7.3f} ${r['mid']:>7.3f} "
                      f"{d_s:>7} {iv_s:>7} {th_s:>7}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Scan NG futures options')
    parser.add_argument('--right', default='C', choices=['C', 'P'],
                        help='C=calls (default), P=puts')
    parser.add_argument('--min-dte', type=int, default=10)
    parser.add_argument('--max-dte', type=int, default=120)
    parser.add_argument('--all-months', action='store_true',
                        help='Scan options on first 4 futures months (not just front)')
    args = parser.parse_args()

    right_label = 'CALL' if args.right == 'C' else 'PUT'

    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=99, timeout=30)

    futs = get_ng_futures(ib)
    print("NG Futures:")
    for c, exp, dte in futs[:8]:
        print(f"  {c.localSymbol:10s}  Exp: {exp}  DTE: {dte}")

    # Determine which futures months to scan options on
    if args.all_months:
        scan_futs = futs[:4]
    else:
        scan_futs = futs[:1]

    all_results = []
    for fut_contract, fut_exp, fut_dte in scan_futs:
        spot = get_spot(ib, fut_contract)
        print(f"\n{'='*70}")
        print(f"Scanning {right_label}S on {fut_contract.localSymbol} "
              f"(${spot:.3f}, exp {fut_exp}, {fut_dte} DTE)")
        print(f"{'='*70}")

        results = scan_options(ib, fut_contract, spot, args.right,
                               args.min_dte, args.max_dte)
        all_results.extend(results)

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"BEST NG {right_label} OPPORTUNITIES")
        print(f"{'='*70}")

        if args.right == 'C':
            # For calls: sort by delta (prefer 0.25-0.40 range), then by IV
            interesting = [r for r in all_results if r['bid'] > 0]
            # Group by bucket
            itm = [r for r in interesting if r['otm_pct'] < 0]
            atm = [r for r in interesting if -2 <= r['otm_pct'] <= 5]
            otm = [r for r in interesting if r['otm_pct'] > 5]

            for label, bucket in [('ATM (Δ~0.50)', atm),
                                  ('OTM (cheap lottery)', otm),
                                  ('ITM (high delta)', itm)]:
                if bucket:
                    print(f"\n  {label}:")
                    bucket.sort(key=lambda x: abs(x.get('delta', 0.5) - 0.35)
                                if x.get('delta') else 99)
                    for r in bucket[:5]:
                        iv_s = f"IV={r['iv']*100:.1f}%" if r['iv'] else ""
                        d_s = f"Δ={r['delta']:.2f}" if r['delta'] else ""
                        th_s = f"θ={r['theta']:.4f}" if r['theta'] else ""
                        notional = r['mid'] * 10000  # NG multiplier
                        print(f"    {r['underlying']} {r['expiry']} "
                              f"${r['strike']:.2f}C ({r['otm_pct']:+.1f}% OTM, {r['dte']}d): "
                              f"bid=${r['bid']:.3f} ask=${r['ask']:.3f} "
                              f"(${notional:,.0f}/ct) {d_s} {iv_s} {th_s}")
        else:
            # For puts: sort by premium yield for selling
            interesting = [r for r in all_results if r['bid'] > 0 and r['otm_pct'] > 0]
            interesting.sort(key=lambda x: (x['bid'] / x['strike']) * (365 / x['dte']),
                             reverse=True)
            print(f"\n  Best puts to SELL (by annualized yield):")
            for r in interesting[:10]:
                ann = (r['bid'] / r['strike']) * (365 / r['dte']) * 100
                iv_s = f"IV={r['iv']*100:.1f}%" if r['iv'] else ""
                d_s = f"Δ={r['delta']:.2f}" if r['delta'] else ""
                notional = r['bid'] * 10000
                print(f"    {r['underlying']} {r['expiry']} "
                      f"${r['strike']:.2f}P ({r['otm_pct']:+.1f}% OTM, {r['dte']}d): "
                      f"bid=${r['bid']:.3f} (${notional:,.0f}/ct, {ann:.1f}% ann) "
                      f"{d_s} {iv_s}")
    else:
        print(f"\nNo {right_label.lower()} data returned.")
        print("If Sunday before 5PM CT, CME Globex hasn't opened yet.")
        print("NG futures trade Sun 5PM CT - Fri 4PM CT.")

    ib.disconnect()


if __name__ == '__main__':
    main()
