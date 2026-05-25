#!/usr/bin/env python3
"""
Gamma Magnet Scanner — LNE (NG futures options) + UNG equity options.

Fetches full option chains (calls + puts) for multiple expirations,
computes gamma exposure (GEX) by strike, and produces a multi-panel chart
showing OI distribution, net GEX, and squeeze/magnet zones.

Usage:
    python gamma_scan.py                  # Scan LNE front 2 months + UNG
    python gamma_scan.py --lne-only       # LNE only
    python gamma_scan.py --ung-only       # UNG only
    python gamma_scan.py --months 4       # LNE front 4 months
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime
from ib_insync import IB, Future, FuturesOption, Stock, Option
from modules.common import IBKR_HOST, IBKR_PORT

MULTIPLIER_NG = 10000   # NG futures: $10k per $1
MULTIPLIER_UNG = 100    # UNG equity: 100 shares per contract


def get_spot(ib, contract, sleep_time=3):
    """Get current price with weekend/holiday fallback."""
    ib.reqMktData(contract)
    ib.sleep(sleep_time)
    t = ib.ticker(contract)
    spot = t.last if t.last and t.last > 0 else t.close
    if not spot or not (spot > 0):
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    return spot


def fetch_lne_chains(ib, num_months=2, min_dte=5, max_dte=120):
    """Fetch LNE option chains for front N futures months, both calls and puts."""
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

    all_data = []
    for fut_contract, fut_exp, fut_dte in futs[:num_months]:
        ib.qualifyContracts(fut_contract)
        spot = get_spot(ib, fut_contract)
        print(f"\nLNE: {fut_contract.localSymbol} spot=${spot:.3f} (exp {fut_exp}, {fut_dte} DTE)")

        opt_params = ib.reqSecDefOptParams(
            fut_contract.symbol, 'NYMEX', 'FUT', fut_contract.conId)
        if not opt_params:
            continue

        # Collect LNE expirations and strikes
        all_exps = set()
        all_strikes = set()
        for op in opt_params:
            if op.tradingClass != 'LNE':
                continue
            for exp in op.expirations:
                exp_dt = datetime.strptime(exp, '%Y%m%d')
                dte = (exp_dt - datetime.now()).days
                if min_dte <= dte <= max_dte:
                    all_exps.add((exp, dte))
            all_strikes.update(op.strikes)

        strikes = sorted([s for s in all_strikes if spot * 0.50 <= s <= spot * 1.80])
        target_exps = sorted(all_exps, key=lambda x: x[1])[:4]  # up to 4 expirations

        print(f"  Strikes: {len(strikes)}, Expirations: {[e[0] for e in target_exps]}")

        for exp, dte in target_exps:
            for right in ['C', 'P']:
                opts = [FuturesOption(symbol='NG', lastTradeDateOrContractMonth=exp,
                                     strike=s, right=right, exchange='NYMEX',
                                     tradingClass='LNE')
                        for s in strikes]
                try:
                    qualified = ib.qualifyContracts(*opts)
                except Exception as e:
                    print(f"    Error qualifying {right} {exp}: {e}")
                    continue

                valid = [o for o in qualified if o.conId > 0]
                if not valid:
                    continue

                for opt in valid:
                    ib.reqMktData(opt, '100,101', False, False)
                ib.sleep(5)

                for opt in valid:
                    t = ib.ticker(opt)
                    bid = t.bid if t.bid and t.bid > 0 else 0
                    ask = t.ask if t.ask and t.ask > 0 else 0
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)

                    delta = t.modelGreeks.delta if t.modelGreeks else None
                    gamma = t.modelGreeks.gamma if t.modelGreeks else None
                    iv = t.modelGreeks.impliedVol if t.modelGreeks else None
                    theta = t.modelGreeks.theta if t.modelGreeks else None

                    # OI: callOpenInterest for calls, putOpenInterest for puts
                    oi = 0
                    if right == 'C':
                        v = getattr(t, 'callOpenInterest', None)
                        if v is not None and v == v and v > 0:  # not nan
                            oi = int(v)
                    else:
                        v = getattr(t, 'putOpenInterest', None)
                        if v is not None and v == v and v > 0:
                            oi = int(v)

                    vol = int(t.volume) if t.volume is not None and t.volume == t.volume and t.volume > 0 else 0

                    all_data.append({
                        'product': 'LNE',
                        'underlying': fut_contract.localSymbol,
                        'spot': spot,
                        'expiry': exp,
                        'dte': dte,
                        'strike': opt.strike,
                        'right': right,
                        'bid': bid, 'ask': ask, 'mid': mid,
                        'delta': delta, 'gamma': gamma, 'iv': iv, 'theta': theta,
                        'oi': oi, 'volume': vol,
                        'multiplier': MULTIPLIER_NG,
                    })

            print(f"    {exp} ({dte}d): fetched C+P")

    return pd.DataFrame(all_data)


def fetch_ung_chains(ib, min_dte=5, max_dte=60):
    """Fetch UNG equity option chains, both calls and puts."""
    stock = Stock('UNG', 'SMART', 'USD')
    ib.qualifyContracts(stock)
    spot = get_spot(ib, stock)
    print(f"\nUNG spot=${spot:.2f}")

    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
    target_exps = []
    all_strikes = set()
    for chain in chains:
        if chain.exchange == 'SMART':
            for exp in chain.expirations:
                exp_dt = datetime.strptime(exp, '%Y%m%d')
                dte = (exp_dt - datetime.now()).days
                if min_dte <= dte <= max_dte:
                    target_exps.append((exp, dte))
            all_strikes.update(chain.strikes)

    target_exps = sorted(set(target_exps), key=lambda x: x[1])[:4]
    strikes = sorted([s for s in all_strikes if spot * 0.60 <= s <= spot * 1.50])
    print(f"  Strikes: {len(strikes)}, Expirations: {[e[0] for e in target_exps]}")

    all_data = []
    for exp, dte in target_exps:
        for right in ['C', 'P']:
            opts = [Option('UNG', exp, s, right, 'SMART') for s in strikes]
            try:
                ib.qualifyContracts(*opts)
            except Exception as e:
                print(f"    Error: {e}")
                continue

            valid = [o for o in opts if o.conId > 0]
            if not valid:
                continue

            for opt in valid:
                ib.reqMktData(opt, '100,101', False, False)
            ib.sleep(4)

            for opt in valid:
                t = ib.ticker(opt)
                bid = t.bid if t.bid and t.bid > 0 else 0
                ask = t.ask if t.ask and t.ask > 0 else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)

                delta = t.modelGreeks.delta if t.modelGreeks else None
                gamma = t.modelGreeks.gamma if t.modelGreeks else None
                iv = t.modelGreeks.impliedVol if t.modelGreeks else None
                theta = t.modelGreeks.theta if t.modelGreeks else None

                oi = 0
                if right == 'C':
                    v = getattr(t, 'callOpenInterest', None)
                    if v is not None and v == v and v > 0:
                        oi = int(v)
                else:
                    v = getattr(t, 'putOpenInterest', None)
                    if v is not None and v == v and v > 0:
                        oi = int(v)

                vol = int(t.volume) if t.volume and t.volume > 0 else 0

                all_data.append({
                    'product': 'UNG',
                    'underlying': 'UNG',
                    'spot': spot,
                    'expiry': exp,
                    'dte': dte,
                    'strike': opt.strike,
                    'right': right,
                    'bid': bid, 'ask': ask, 'mid': mid,
                    'delta': delta, 'gamma': gamma, 'iv': iv, 'theta': theta,
                    'oi': oi, 'volume': vol,
                    'multiplier': MULTIPLIER_UNG,
                })

        print(f"    {exp} ({dte}d): fetched C+P")

    return pd.DataFrame(all_data)


def compute_gex(df):
    """Compute Gamma Exposure (GEX) per strike.

    GEX = OI * Gamma * Spot^2 * Multiplier * 0.01
    Convention: call gamma is positive, put gamma is negative (dealer short options).
    """
    if df.empty:
        return df

    df = df.copy()
    df['abs_gamma'] = df['gamma'].abs().fillna(0)
    # Dealer is typically short options → short call gamma (negative), short put gamma (positive for dealer)
    # Call GEX: negative (dealer short calls → short gamma → sells on rally)
    # Put GEX: positive (dealer short puts → long gamma effect inverted → buys on decline)
    # Net: positive = dealer buys dips (magnet), negative = dealer sells rallies (resistance)
    # Simplified: GEX = OI * |gamma| * spot^2 * mult * 0.01, sign = +1 for calls, -1 for puts
    df['gex'] = df.apply(
        lambda r: r['oi'] * r['abs_gamma'] * r['spot']**2 * r['multiplier'] * 0.01 *
                  (1 if r['right'] == 'C' else -1),
        axis=1
    )
    return df


def make_chart(df, output_path):
    """Create multi-panel gamma exposure chart with OI + GEX side by side."""
    products = df['product'].unique()
    # Group by product + underlying, pick top 2 expirations by OI
    groups = []
    for prod in sorted(products):
        sub = df[df['product'] == prod]
        for und in sorted(sub['underlying'].unique()):
            usub = sub[sub['underlying'] == und]
            exps = usub.groupby('expiry')['oi'].sum().sort_values(ascending=False)
            top_exps = exps.head(2).index.tolist()
            for exp in sorted(top_exps):
                edata = usub[usub['expiry'] == exp]
                dte = edata['dte'].iloc[0]
                spot = edata['spot'].iloc[0]
                groups.append((prod, und, exp, dte, spot, edata))

    n_panels = len(groups)
    if n_panels == 0:
        print("No data to chart.")
        return

    fig = plt.figure(figsize=(24, 7 * n_panels), facecolor='#0d1117')
    gs = gridspec.GridSpec(n_panels, 2, width_ratios=[1, 1],
                           hspace=0.30, wspace=0.20,
                           left=0.05, right=0.97, top=0.94, bottom=0.03)

    C_BG = '#0d1117'
    C_PANEL = '#161b22'
    C_TEXT = '#e6edf3'
    C_GRID = '#21262d'
    C_CALL = '#3fb950'
    C_PUT = '#f85149'
    C_GEX_NEG = '#da3633'
    C_SPOT = '#f0883e'
    C_MAGNET = '#ffd700'

    for i, (prod, und, exp, dte, spot, edata) in enumerate(groups):
        calls = edata[edata['right'] == 'C'].groupby('strike').agg(
            oi=('oi', 'sum'), volume=('volume', 'sum'), delta=('delta', 'first'),
            iv=('iv', 'first'), bid=('bid', 'first'), ask=('ask', 'first')
        ).reset_index()
        puts = edata[edata['right'] == 'P'].groupby('strike').agg(
            oi=('oi', 'sum'), volume=('volume', 'sum'), delta=('delta', 'first'),
            iv=('iv', 'first'), bid=('bid', 'first'), ask=('ask', 'first')
        ).reset_index()

        bar_w = (calls['strike'].diff().median() if len(calls) > 1 else 0.25) * 0.35

        # Zoom range: ±40% around spot for LNE, ±50% for UNG
        zoom = 0.40 if prod == 'LNE' else 0.50
        x_lo, x_hi = spot * (1 - zoom), spot * (1 + zoom)

        # ---- Panel 1: OI by strike (calls vs puts) ----
        ax_oi = fig.add_subplot(gs[i, 0])
        ax_oi.set_facecolor(C_PANEL)

        if not calls.empty:
            ax_oi.bar(calls['strike'] - bar_w/2, calls['oi'], width=bar_w,
                      color=C_CALL, alpha=0.85, label='Call OI', edgecolor='none')
        if not puts.empty:
            ax_oi.bar(puts['strike'] + bar_w/2, puts['oi'], width=bar_w,
                      color=C_PUT, alpha=0.85, label='Put OI', edgecolor='none')

        ax_oi.axvline(spot, color=C_SPOT, linewidth=2.5, linestyle='--', alpha=0.9,
                      label=f'Spot ${spot:.2f}')

        # Max OI strike
        all_oi = pd.concat([
            calls[['strike', 'oi']].assign(side='C'),
            puts[['strike', 'oi']].assign(side='P')
        ])
        total_by_strike = all_oi.groupby('strike')['oi'].sum()
        if not total_by_strike.empty and total_by_strike.max() > 0:
            max_oi_strike = total_by_strike.idxmax()
            max_oi_val = total_by_strike.max()
            ax_oi.axvline(max_oi_strike, color=C_MAGNET, linewidth=2, linestyle=':',
                          alpha=0.8, label=f'Max OI ${max_oi_strike:.2f} ({max_oi_val:,.0f})')

        # P/C ratio annotation
        total_c = calls['oi'].sum()
        total_p = puts['oi'].sum()
        pc_all = total_p / total_c if total_c > 0 else float('inf')

        # Top OI table inset
        top_strikes = total_by_strike.sort_values(ascending=False).head(8)
        tbl_lines = ['Top Strikes by OI:']
        for strike, oi_val in top_strikes.items():
            c_oi = calls.loc[calls['strike'] == strike, 'oi'].sum()
            p_oi = puts.loc[puts['strike'] == strike, 'oi'].sum()
            pct = (strike - spot) / spot * 100
            pc = f"{p_oi/c_oi:.1f}" if c_oi > 0 else "inf"
            tbl_lines.append(f"  ${strike:.2f} ({pct:+.0f}%): {oi_val:>7,.0f}  C:{c_oi:>6,.0f} P:{p_oi:>6,.0f}  P/C={pc}")
        tbl_lines.append(f"\nTotal C={total_c:,.0f}  P={total_p:,.0f}  P/C={pc_all:.2f}")

        ax_oi.text(0.98, 0.97, '\n'.join(tbl_lines), transform=ax_oi.transAxes,
                   fontsize=8, fontfamily='monospace', color=C_TEXT,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=C_BG,
                             edgecolor=C_GRID, alpha=0.92))

        ax_oi.set_xlim(x_lo, x_hi)
        ax_oi.set_title(f'{prod} {und} — {exp} ({dte}d) — Open Interest by Strike',
                        color=C_TEXT, fontsize=13, fontweight='bold')
        ax_oi.set_xlabel('Strike', color=C_TEXT, fontsize=11)
        ax_oi.set_ylabel('Open Interest', color=C_TEXT, fontsize=11)
        ax_oi.legend(loc='upper left', fontsize=9, facecolor=C_PANEL, edgecolor=C_GRID,
                     labelcolor=C_TEXT)
        ax_oi.tick_params(colors=C_TEXT, labelsize=10)
        ax_oi.grid(True, alpha=0.15, color=C_GRID)
        for spine in ax_oi.spines.values():
            spine.set_color(C_GRID)

        # ---- Panel 2: Net GEX by strike ----
        ax_gex = fig.add_subplot(gs[i, 1])
        ax_gex.set_facecolor(C_PANEL)

        gex_data = compute_gex(edata)
        gex_by_strike = gex_data.groupby('strike')['gex'].sum().reset_index()
        gex_by_strike = gex_by_strike.sort_values('strike')

        # Separate call and put GEX for stacked display
        call_gex = gex_data[gex_data['right'] == 'C'].groupby('strike')['gex'].sum()
        put_gex = gex_data[gex_data['right'] == 'P'].groupby('strike')['gex'].sum()

        bar_w2 = bar_w * 1.8
        # Stacked bars: call GEX (green/positive) and put GEX (red/negative)
        for strike in gex_by_strike['strike']:
            cg = call_gex.get(strike, 0) / 1e6
            pg = put_gex.get(strike, 0) / 1e6
            if cg > 0:
                ax_gex.bar(strike, cg, width=bar_w2, color=C_CALL, alpha=0.7, edgecolor='none')
            elif cg < 0:
                ax_gex.bar(strike, cg, width=bar_w2, color=C_CALL, alpha=0.4, edgecolor='none')
            if pg < 0:
                ax_gex.bar(strike, pg, width=bar_w2, color=C_PUT, alpha=0.7,
                           bottom=min(cg, 0), edgecolor='none')
            elif pg > 0:
                ax_gex.bar(strike, pg, width=bar_w2, color=C_PUT, alpha=0.4,
                           bottom=max(cg, 0), edgecolor='none')

        # Net GEX line
        ax_gex.plot(gex_by_strike['strike'], gex_by_strike['gex'] / 1e6,
                    color='#58a6ff', linewidth=2, alpha=0.9, marker='o', markersize=3,
                    label='Net GEX')

        ax_gex.axhline(0, color=C_TEXT, linewidth=0.8, alpha=0.4)
        ax_gex.axvline(spot, color=C_SPOT, linewidth=2.5, linestyle='--', alpha=0.9,
                       label=f'Spot ${spot:.2f}')

        # Gamma flip points
        gex_vals = gex_by_strike['gex'].values
        gex_strikes = gex_by_strike['strike'].values
        flip_points = []
        for j in range(1, len(gex_vals)):
            if gex_vals[j-1] * gex_vals[j] < 0:
                flip = (gex_strikes[j-1] + gex_strikes[j]) / 2
                flip_points.append(flip)
                ax_gex.axvline(flip, color='#a371f7', linewidth=2, linestyle='-.',
                               alpha=0.7, label=f'Flip ${flip:.2f}' if j == 1 else '')

        # Annotate flip points
        ylims = ax_gex.get_ylim()
        for fp in flip_points:
            if x_lo <= fp <= x_hi:
                ax_gex.annotate(f'FLIP ${fp:.2f}', xy=(fp, ylims[1] * 0.88),
                                fontsize=9, color='#a371f7', ha='center', fontweight='bold',
                                rotation=90)

        # Mark strongest magnet and squeeze
        if not gex_by_strike.empty:
            # Strongest magnet (most positive)
            pos_gex = gex_by_strike[gex_by_strike['gex'] > 0]
            if not pos_gex.empty:
                max_pos = pos_gex.loc[pos_gex['gex'].idxmax()]
                ax_gex.annotate(f'MAGNET ${max_pos["strike"]:.2f}\n(${max_pos["gex"]/1e6:.1f}M)',
                                xy=(max_pos['strike'], max_pos['gex'] / 1e6),
                                xytext=(0, 15), textcoords='offset points',
                                fontsize=9, color=C_MAGNET, ha='center', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color=C_MAGNET, lw=1.5))
            # Strongest squeeze (most negative)
            neg_gex = gex_by_strike[gex_by_strike['gex'] < 0]
            if not neg_gex.empty:
                min_neg = neg_gex.loc[neg_gex['gex'].idxmin()]
                ax_gex.annotate(f'SQUEEZE ${min_neg["strike"]:.2f}\n(${min_neg["gex"]/1e6:.1f}M)',
                                xy=(min_neg['strike'], min_neg['gex'] / 1e6),
                                xytext=(0, -20), textcoords='offset points',
                                fontsize=9, color=C_GEX_NEG, ha='center', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color=C_GEX_NEG, lw=1.5))

        # Shade squeeze zone
        neg_gex_df = gex_by_strike[gex_by_strike['gex'] < 0]
        if not neg_gex_df.empty:
            thresh = neg_gex_df['gex'].quantile(0.25)
            for _, row in neg_gex_df[neg_gex_df['gex'] <= thresh].iterrows():
                ax_gex.axvspan(row['strike'] - bar_w, row['strike'] + bar_w,
                               alpha=0.12, color=C_GEX_NEG)

        # GEX summary inset
        gex_total = gex_by_strike['gex'].sum()
        regime = 'POSITIVE — Magnet dominant' if gex_total > 0 else 'NEGATIVE — Squeeze dominant'
        gex_lines = [
            f'Net GEX: ${gex_total/1e6:.1f}M — {regime}',
            f'Call GEX: ${call_gex.sum()/1e6:.1f}M  Put GEX: ${put_gex.sum()/1e6:.1f}M',
        ]
        if flip_points:
            gex_lines.append(f'Flip point(s): ' + ', '.join(f'${f:.2f}' for f in flip_points))
        gex_lines.append('')
        gex_lines.append('Green bars=Call GEX  Red bars=Put GEX')
        gex_lines.append('Blue line=Net GEX  Purple=Gamma flip')

        ax_gex.text(0.02, 0.97, '\n'.join(gex_lines), transform=ax_gex.transAxes,
                    fontsize=8.5, fontfamily='monospace', color=C_TEXT,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=C_BG,
                              edgecolor=C_GRID, alpha=0.92))

        ax_gex.set_xlim(x_lo, x_hi)
        ax_gex.set_title(f'Net Gamma Exposure (GEX) — Magnets & Squeeze Zones',
                         color=C_TEXT, fontsize=13, fontweight='bold')
        ax_gex.set_xlabel('Strike', color=C_TEXT, fontsize=11)
        ax_gex.set_ylabel('GEX ($M)', color=C_TEXT, fontsize=11)
        ax_gex.tick_params(colors=C_TEXT, labelsize=10)
        ax_gex.grid(True, alpha=0.15, color=C_GRID)
        for spine in ax_gex.spines.values():
            spine.set_color(C_GRID)

    fig.suptitle(f'NG Gamma Magnet Scanner — {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 color=C_TEXT, fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, facecolor=C_BG, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='NG Gamma Magnet Scanner')
    parser.add_argument('--lne-only', action='store_true', help='LNE only')
    parser.add_argument('--ung-only', action='store_true', help='UNG only')
    parser.add_argument('--months', type=int, default=2, help='LNE futures months (default: 2)')
    parser.add_argument('--output', default='gamma_scan.png', help='Output file')
    args = parser.parse_args()

    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=97, timeout=30)

    frames = []

    if not args.ung_only:
        lne_df = fetch_lne_chains(ib, num_months=args.months)
        if not lne_df.empty:
            frames.append(lne_df)
            print(f"\nLNE: {len(lne_df)} option rows fetched")

    if not args.lne_only:
        ung_df = fetch_ung_chains(ib)
        if not ung_df.empty:
            frames.append(ung_df)
            print(f"UNG: {len(ung_df)} option rows fetched")

    ib.disconnect()

    if frames:
        df = pd.concat(frames, ignore_index=True)
        print(f"\nTotal: {len(df)} rows, {df['oi'].sum():,} total OI")

        # Console summary
        for prod in df['product'].unique():
            psub = df[df['product'] == prod]
            for und in psub['underlying'].unique():
                usub = psub[psub['underlying'] == und]
                for exp in sorted(usub['expiry'].unique()):
                    esub = usub[usub['expiry'] == exp]
                    dte = esub['dte'].iloc[0]
                    spot = esub['spot'].iloc[0]
                    gex = compute_gex(esub)
                    gex_by_s = gex.groupby('strike')['gex'].sum().sort_values(ascending=False)

                    print(f"\n{'='*70}")
                    print(f"{prod} {und} — {exp} ({dte}d) — Spot ${spot:.2f}")
                    print(f"{'='*70}")

                    # Top 5 gamma magnets
                    print("  TOP GAMMA MAGNETS (positive GEX = price attractor):")
                    for strike, gval in gex_by_s.head(5).items():
                        pct = (strike - spot) / spot * 100
                        c_oi = esub[(esub['strike'] == strike) & (esub['right'] == 'C')]['oi'].sum()
                        p_oi = esub[(esub['strike'] == strike) & (esub['right'] == 'P')]['oi'].sum()
                        print(f"    ${strike:.2f} ({pct:+.1f}%): GEX=${gval/1e6:.1f}M  "
                              f"C_OI={c_oi:,} P_OI={p_oi:,}")

                    # Squeeze zones (negative GEX)
                    neg = gex_by_s[gex_by_s < 0].sort_values()
                    if not neg.empty:
                        print("  SQUEEZE ZONES (negative GEX = amplifies moves):")
                        for strike, gval in neg.head(5).items():
                            pct = (strike - spot) / spot * 100
                            print(f"    ${strike:.2f} ({pct:+.1f}%): GEX=${gval/1e6:.1f}M")

        make_chart(df, args.output)
    else:
        print("No data fetched.")


if __name__ == '__main__':
    main()
