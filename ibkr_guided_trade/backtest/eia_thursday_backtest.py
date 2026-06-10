"""EIA Thursday vol-crush strategy: focused backtest.

Trade structure:
  WED close: sell ATM short strangle (~7 DTE) — capture inflated pre-event IV
  THU release at 10:30 ET: IV crushes — position gains intrinsic-zero
  FRI close: BTC remaining premium (now small)

Pessimism modeling (per user request):
  Wed open spread:  2× normal (event-anticipation widening)
  Fri close spread: 1.5× normal (post-event normalization but still wider than calm)

Position sizing:
  Cap exposure at 5% NAV per event = avoid wipeout on rare bad weeks
  Use real strikes from PG (ATM ± 0.5)
"""
from __future__ import annotations
import os
import sys
import math
import pandas as pd
import numpy as np
import psycopg2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DB = dict(host='192.168.1.172', port=5432, database='market_scanner',
          user='postgres', password=os.environ.get('SHINOBI_PG_PASSWORD', ''), connect_timeout=8)

# Cost model
SPREAD_OPTION_NORMAL = 0.05    # $0.05 half-spread (matches replay_engine)
SPREAD_OPEN_MULT = 2.0          # pre-event spread widening (user's caveat)
SPREAD_CLOSE_MULT = 1.5         # post-event spreads still wider than normal
COMMISSION_PER_CONTRACT = 0.65


def load_iv_pg():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT date, expiration, dte, strike_real, option_right,
               spot_real, mid, iv
        FROM ung_iv_surface
        WHERE dte BETWEEN 1 AND 14
        ORDER BY date, expiration, strike_real
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['expiration'] = pd.to_datetime(df['expiration']).dt.normalize()
    return df


def get_atm_strikes(df_iv, date, target_otm=0.02):
    """Return (put_strike, call_strike) for ATM short strangle on this date.
    Use ~2% OTM each side."""
    day = df_iv[df_iv['date'] == date]
    if day.empty:
        return None, None, None, None
    spot = float(day.iloc[0]['spot_real'])
    # Want next-Friday expiry (~ 7-9 DTE on Wed)
    exp_options = day[(day['dte'] >= 5) & (day['dte'] <= 14)]
    if exp_options.empty:
        return None, None, None, None
    target_exp = exp_options['expiration'].min()
    # Pick put strike ~2% OTM (below spot), call strike ~2% OTM (above spot)
    target_put = spot * (1 - target_otm)
    target_call = spot * (1 + target_otm)
    leg_put = day[(day['expiration'] == target_exp) & (day['option_right'] == 'P')]
    leg_call = day[(day['expiration'] == target_exp) & (day['option_right'] == 'C')]
    if leg_put.empty or leg_call.empty:
        return None, None, None, None
    put_row = leg_put.iloc[(leg_put['strike_real'] - target_put).abs().argsort()[:1]].iloc[0]
    call_row = leg_call.iloc[(leg_call['strike_real'] - target_call).abs().argsort()[:1]].iloc[0]
    return float(put_row['strike_real']), float(call_row['strike_real']), put_row, call_row


def get_premium(df_iv, date, expiration, strike, right):
    sel = df_iv[(df_iv['date'] == date) &
                (df_iv['expiration'] == expiration) &
                (df_iv['strike_real'] == strike) &
                (df_iv['option_right'] == right)]
    if sel.empty:
        return None
    return float(sel.iloc[0]['mid'])


def run_backtest(nav_start=100000, max_pct_per_event=0.05):
    df_iv = load_iv_pg()
    # All available dates
    dates = sorted(df_iv['date'].unique())
    print(f'Loaded {len(dates)} unique IV dates')

    # Find all Wed-Fri pairs (Wed of week N, Fri of same week)
    pairs = []
    for d in dates:
        if d.weekday() == 2:  # Wednesday
            fri_target = d + pd.Timedelta(days=2)
            if fri_target in set(dates):
                pairs.append((d, fri_target))
    print(f'Wed-Fri pairs: {len(pairs)}')

    nav = nav_start
    history = [{'date': pairs[0][0], 'nav': nav, 'event_pnl': 0}]
    events = []
    for wed, fri in pairs:
        K_p, K_c, p_row, c_row = get_atm_strikes(df_iv, wed)
        if K_p is None:
            continue
        exp = pd.Timestamp(p_row['expiration'])
        put_open = float(p_row['mid'])
        call_open = float(c_row['mid'])
        if put_open <= 0.10 or call_open <= 0.10:
            continue  # skip ultra-cheap (no edge to capture)

        # SIZING: max collateral = max_pct_per_event * nav, split between put + call
        max_collateral = nav * max_pct_per_event
        put_collateral_per = K_p * 100
        call_collateral_per = K_c * 100  # naked-call equivalent; we'd actually need shares
        qty = max(1, int(max_collateral / max(put_collateral_per, call_collateral_per) / 2))
        qty = min(qty, 5)  # cap

        # Open credit (Wed close, wide spreads)
        eff_open_spread = SPREAD_OPTION_NORMAL * SPREAD_OPEN_MULT
        credit_put = (put_open - eff_open_spread) * 100 * qty
        credit_call = (call_open - eff_open_spread) * 100 * qty
        gross_credit = credit_put + credit_call - 4 * qty * COMMISSION_PER_CONTRACT

        # Close (Fri) — look up Fri mid for same strikes/expiry
        put_close = get_premium(df_iv, fri, exp, K_p, 'P')
        call_close = get_premium(df_iv, fri, exp, K_c, 'C')
        if put_close is None or call_close is None:
            # Position expires/assigns by spot at fri
            day_fri = df_iv[df_iv['date'] == fri]
            if day_fri.empty: continue
            spot_fri = float(day_fri.iloc[0]['spot_real'])
            put_close = max(0, K_p - spot_fri)
            call_close = max(0, spot_fri - K_c)
        eff_close_spread = SPREAD_OPTION_NORMAL * SPREAD_CLOSE_MULT
        debit_put = (put_close + eff_close_spread) * 100 * qty
        debit_call = (call_close + eff_close_spread) * 100 * qty
        gross_debit = debit_put + debit_call + 4 * qty * COMMISSION_PER_CONTRACT

        pnl = gross_credit - gross_debit
        nav += pnl
        history.append({'date': fri, 'nav': nav, 'event_pnl': pnl})
        events.append({
            'wed': wed.date(), 'fri': fri.date(), 'spot': float(p_row['spot_real']),
            'put_K': K_p, 'call_K': K_c, 'qty': qty,
            'credit': gross_credit, 'debit': gross_debit, 'pnl': pnl,
            'iv_open_p': float(p_row['iv']), 'iv_open_c': float(c_row['iv']),
        })

    hist_df = pd.DataFrame(history).set_index('date')
    hist_df['nav_ret'] = hist_df['nav'].pct_change()
    rets = hist_df['nav_ret'].dropna()
    years = (hist_df.index[-1] - hist_df.index[0]).days / 365.25
    total_ret = (nav / nav_start - 1) * 100
    ann_ret = (nav / nav_start) ** (1 / max(years, 0.1)) * 100 - 100
    sharpe = rets.mean() / (rets.std() + 1e-9) * math.sqrt(52)  # weekly events
    win_rate = (rets > 0).sum() / len(rets) * 100 if len(rets) > 0 else 0
    peak = hist_df['nav'].cummax()
    mdd = ((hist_df['nav'] - peak) / peak * 100).min()

    print()
    print(f'═══════ EIA Thursday Vol-Crush Results (with pessimistic spread) ═══════')
    print(f'  Events: {len(events)}  Years: {years:.1f}')
    print(f'  NAV: ${nav_start:,.0f} → ${nav:,.0f}  (total {total_ret:+.1f}%)')
    print(f'  Annualized: {ann_ret:+.1f}%')
    print(f'  Sharpe (weekly): {sharpe:+.2f}')
    print(f'  Win rate: {win_rate:.0f}%')
    print(f'  Max DD: {mdd:+.1f}%')
    print()
    events_df = pd.DataFrame(events)
    print(f'  P&L distribution:')
    print(f'    Best week: ${events_df["pnl"].max():+,.0f}')
    print(f'    Worst week: ${events_df["pnl"].min():+,.0f}')
    print(f'    Avg P&L: ${events_df["pnl"].mean():+,.0f}')
    print(f'    Median P&L: ${events_df["pnl"].median():+,.0f}')
    print(f'    Std P&L: ${events_df["pnl"].std():,.0f}')
    print()
    losers = events_df[events_df['pnl'] < -500]
    print(f'  Bad weeks (>$500 loss): {len(losers)} ({len(losers)/len(events_df)*100:.0f}%)')
    if not losers.empty:
        print(f'    Sample worst:')
        for _, r in losers.nsmallest(5, 'pnl').iterrows():
            print(f'      {r["wed"]} → {r["fri"]}: spot {r["spot"]:.2f}, qty {r["qty"]}, pnl ${r["pnl"]:+,.0f}')
    return hist_df, events_df


if __name__ == '__main__':
    run_backtest()
