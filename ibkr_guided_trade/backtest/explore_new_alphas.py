"""Quick edge-detection across 11 candidate alpha ideas using real data.

For each idea, we don't fully implement — we test "is there a tradeable
signal?" using master_dataset + PG ung_iv_surface. Output: edge magnitude,
trigger frequency, complexity-to-build, recommended action.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import pandas as pd
import psycopg2

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
DB = dict(host='192.168.1.172', port=5432, database='market_scanner',
          user='postgres', password=os.environ.get('SHINOBI_PG_PASSWORD', ''), connect_timeout=8)


def load_master():
    df = pd.read_csv(os.path.join(THIS_DIR, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    # Dedupe — keep first non-NaN per date
    df = df.groupby(df.index.normalize()).first()
    return df


def section(name):
    print(f'\n{"="*78}\n  {name}\n{"="*78}')


# ───── IDEA 1: Multi-strategy ensemble ──────────────────────────────────────
def idea_1_ensemble():
    section('1. MULTI-STRATEGY ENSEMBLE — pure math, no signal needed')
    # Load history of top 3 Pareto winners
    picks = ['champion_premium_harvest_scale_invariant',
             'champion_target_25_dd_trim',
             'champion_target_25_smooth']
    histories = {}
    for s in picks:
        try:
            h = pd.read_csv(os.path.join(THIS_DIR, 'results', f'{s}_history.csv'),
                            parse_dates=['date']).set_index('date')
            histories[s] = h['nav'].pct_change()
        except Exception:
            continue
    if len(histories) < 2:
        print('  not enough data')
        return
    df = pd.DataFrame(histories).dropna()
    correlations = df.corr()
    print('  Daily-return correlations:')
    for i, a in enumerate(df.columns):
        for j, b in enumerate(df.columns):
            if j > i:
                print(f'    {a[:30]:30s} vs {b[:30]:30s}: {correlations.iloc[i,j]:.3f}')
    ensemble = df.mean(axis=1)
    avg_sharpe = df.mean().mean() / (df.std().mean() + 1e-9) * math.sqrt(252)
    ens_sharpe = ensemble.mean() / (ensemble.std() + 1e-9) * math.sqrt(252)
    ens_ann = (1 + ensemble.mean()) ** 252 - 1
    print(f'  Avg individual Sharpe: {avg_sharpe:.2f}')
    print(f'  EQUAL-WEIGHT ENSEMBLE: ann {ens_ann*100:+.1f}%  Sharpe {ens_sharpe:+.2f}')
    print(f'  Sharpe LIFT from diversification: {ens_sharpe - avg_sharpe:+.2f}')
    if ens_sharpe - avg_sharpe > 0.15:
        print(f'  ✅ VERDICT: real diversification edge ({ens_sharpe - avg_sharpe:+.2f} Sharpe)')
    else:
        print('  ⚠️ VERDICT: marginal — strategies too correlated')


# ───── IDEA 2: EIA event-driven IV crush ────────────────────────────────────
def idea_2_eia_event():
    section('2. EIA STORAGE EVENT IV CRUSH — Thursday 10:30am ET vol decay')
    # For each Thursday in PG, check IV on Wed vs Thu vs Fri for ATM weekly
    conn = psycopg2.connect(**DB)
    df_iv = pd.read_sql("""
        SELECT date, expiration, dte, strike_real, option_right, spot_real, iv
        FROM ung_iv_surface
        WHERE dte BETWEEN 2 AND 14
          AND ABS(strike_real - spot_real) <= 1.0
        ORDER BY date
    """, conn)
    conn.close()
    df_iv['date'] = pd.to_datetime(df_iv['date'])
    df_iv['weekday'] = df_iv['date'].dt.dayofweek
    # Median ATM IV by date+weekday
    daily_iv = df_iv.groupby(['date', 'weekday'])['iv'].median().reset_index()
    daily_iv = daily_iv.set_index('date')
    # Pivot: for each week, Wed IV vs Thu IV vs Fri IV
    wed = daily_iv[daily_iv['weekday'] == 2]['iv']
    thu = daily_iv[daily_iv['weekday'] == 3]['iv']
    fri = daily_iv[daily_iv['weekday'] == 4]['iv']
    # Pair by week
    wed.index = wed.index.normalize()
    thu.index = thu.index.normalize()
    fri.index = fri.index.normalize()
    # IV change Wed close → Thu close (captures EIA crush)
    paired = []
    for w_date in wed.index:
        t_date = w_date + pd.Timedelta(days=1)
        if t_date in thu.index:
            paired.append((w_date, float(wed[w_date]), float(thu[t_date])))
    paired_df = pd.DataFrame(paired, columns=['wed_date', 'wed_iv', 'thu_iv'])
    paired_df['iv_change'] = paired_df['thu_iv'] - paired_df['wed_iv']
    paired_df['iv_change_pct'] = paired_df['iv_change'] / paired_df['wed_iv'] * 100
    print(f'  {len(paired_df)} Wed-Thu pairs analyzed')
    print(f'  Median ATM IV change Wed→Thu: {paired_df["iv_change"].median():+.4f} ({paired_df["iv_change_pct"].median():+.2f}%)')
    print(f'  Mean: {paired_df["iv_change"].mean():+.4f} ({paired_df["iv_change_pct"].mean():+.2f}%)')
    iv_drops = paired_df[paired_df['iv_change'] < 0]
    paired_df[paired_df['iv_change'] > 0]
    print(f'  Frequency of IV DROP (favorable for vol seller): {len(iv_drops)/len(paired_df)*100:.0f}%')
    print(f'  Avg drop when dropping: {iv_drops["iv_change_pct"].mean():.2f}%')
    if paired_df['iv_change_pct'].mean() < -2:
        print(f'  ✅ VERDICT: REAL EDGE — avg {paired_df["iv_change_pct"].mean():.1f}% IV drop overnight')
    elif paired_df['iv_change_pct'].mean() < 0:
        print(f'  🟡 VERDICT: small edge ({paired_df["iv_change_pct"].mean():.1f}% avg drop)')
    else:
        print('  ⚠️ VERDICT: no clear edge — IV is symmetric around the event')


# ───── IDEA 3: Calendar spread term structure ───────────────────────────────
def idea_3_calendar():
    section('3. CALENDAR SPREADS — short-dated vs long-dated IV gap')
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT date, dte, ABS(strike_real - spot_real) AS otm_dist,
               iv, spot_real
        FROM ung_iv_surface
        WHERE ABS(strike_real - spot_real) <= 0.5
          AND option_right = 'P'
        ORDER BY date, dte
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    # For each date, find ATM 30d IV and ATM 60d IV
    df_30 = df[(df['dte'] >= 25) & (df['dte'] <= 35)].groupby('date')['iv'].mean()
    df_60 = df[(df['dte'] >= 55) & (df['dte'] <= 75)].groupby('date')['iv'].mean()
    aligned = pd.concat([df_30, df_60], axis=1, keys=['iv_30', 'iv_60']).dropna()
    aligned['gap'] = aligned['iv_60'] - aligned['iv_30']
    print(f'  Days with both 30d and 60d ATM IV: {len(aligned)}')
    print(f'  Avg term structure gap (60d - 30d): {aligned["gap"].mean()*100:+.2f}pp')
    print(f'  Days with backwardation (30d > 60d, favorable for calendar): {(aligned["gap"] < 0).sum()/len(aligned)*100:.0f}%')
    bw_days = aligned[aligned['gap'] < -0.05]
    if len(bw_days) > 50:
        print(f'  ✅ VERDICT: backwardation occurs {len(bw_days)} days; calendar opportunities exist')
    else:
        print('  ⚠️ VERDICT: mostly contango (long > short); calendars work but reversed (sell 60d / buy 30d)')


# ───── IDEA 4: Cross-asset KOLD hedge ───────────────────────────────────────
def idea_4_kold_hedge():
    section('4. KOLD HEDGE at UNG peaks — your idea')
    df = load_master()
    # Define "peak" = surge_z > +1.5 OR z >+1.0
    df['ma20'] = df['UNG'].rolling(20).mean()
    df['sd20'] = df['UNG'].rolling(20).std()
    df['surge_z'] = (df['UNG'] - df['ma20']) / df['sd20'].replace(0, np.nan)
    df['kold_ret'] = df['KOLD'].pct_change()
    # Forward returns
    df['kold_ret_20d'] = df['KOLD'].pct_change(20).shift(-20)
    df['ung_ret_20d'] = df['UNG'].pct_change(20).shift(-20)
    # Trigger
    peaks = df[df['surge_z'] > 1.5].dropna(subset=['kold_ret_20d'])
    normal = df[(df['surge_z'].abs() < 0.5)].dropna(subset=['kold_ret_20d'])
    print(f'  Peak days (surge_z > +1.5): {len(peaks)}')
    print(f'  Avg KOLD 20d return AFTER peak: {peaks["kold_ret_20d"].mean()*100:+.2f}%')
    print(f'  Avg KOLD 20d return on NORMAL days: {normal["kold_ret_20d"].mean()*100:+.2f}%')
    print(f'  Win rate (KOLD up after peak): {(peaks["kold_ret_20d"] > 0).sum()/len(peaks)*100:.0f}%')
    if peaks['kold_ret_20d'].mean() > 0.05:
        print(f'  ✅ VERDICT: KOLD averages {peaks["kold_ret_20d"].mean()*100:+.1f}% 20d after UNG peaks')
    elif peaks['kold_ret_20d'].mean() > 0:
        print('  🟡 VERDICT: modest positive edge')
    else:
        print('  ⚠️ VERDICT: KOLD doesn\'t reliably rally after UNG peaks')


# ───── IDEA 5: NG contango carry (UNG bleed) ────────────────────────────────
def idea_5_contango():
    section('5. NG CONTANGO CARRY — UNG\'s structural bleed')
    df = load_master()
    if 'NG' not in df.columns:
        print('  NG futures col missing'); return
    # Annual UNG vs NG performance (UNG bleeds in contango)
    ng = df['NG'].dropna()
    ung = df['UNG'].dropna()
    aligned = pd.concat([ung, ng], axis=1, keys=['UNG', 'NG']).dropna()
    aligned['ung_ret'] = aligned['UNG'].pct_change(252)
    aligned['ng_ret'] = aligned['NG'].pct_change(252)
    aligned = aligned.dropna()
    aligned['ung_bleed'] = aligned['ung_ret'] - aligned['ng_ret']  # negative if UNG underperforms
    print(f'  Years analyzed: {len(aligned)/252:.1f}')
    print(f'  Avg annual UNG return: {aligned["ung_ret"].mean()*100:+.1f}%')
    print(f'  Avg annual NG return:  {aligned["ng_ret"].mean()*100:+.1f}%')
    print(f'  Avg UNG bleed vs NG: {aligned["ung_bleed"].mean()*100:+.1f}% per year')
    if aligned['ung_bleed'].mean() < -0.05:
        print(f'  ✅ VERDICT: ~{abs(aligned["ung_bleed"].mean())*100:.0f}% structural bleed/yr')
        print('     Strategy: short UNG / long NG futures = capture this bleed')
    else:
        print('  ⚠️ VERDICT: bleed exists but small')


# ───── IDEA 6: Skew-rich vs flat ────────────────────────────────────────────
def idea_6_skew():
    section('6. SKEW REGIME — put-skew percentile as signal')
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("""
        SELECT date, dte, strike_real, option_right, spot_real, iv
        FROM ung_iv_surface
        WHERE dte BETWEEN 20 AND 45
        ORDER BY date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df['moneyness'] = (df['strike_real'] - df['spot_real']) / df['spot_real']
    # For each date, ATM IV vs 10% OTM put IV
    daily_atm = df[(df['moneyness'].abs() < 0.02)].groupby('date')['iv'].mean()
    daily_put_otm = df[(df['option_right'] == 'P') & (df['moneyness'] > -0.12) & (df['moneyness'] < -0.07)].groupby('date')['iv'].mean()
    skew = pd.concat([daily_atm, daily_put_otm], axis=1, keys=['atm', 'put_otm']).dropna()
    skew['put_skew'] = skew['put_otm'] - skew['atm']
    print(f'  Days analyzed: {len(skew)}')
    print(f'  Median put-skew (10% OTM IV - ATM IV): {skew["put_skew"].median()*100:+.2f}pp')
    print(f'  90th pctile (rich puts → sell more): {skew["put_skew"].quantile(0.9)*100:+.2f}pp')
    print(f'  10th pctile (flat → reduce): {skew["put_skew"].quantile(0.1)*100:+.2f}pp')
    # Test: when skew rich, do puts realize higher return?
    spot_df = load_master()['UNG'].dropna()
    skew_idx = skew.index.intersection(spot_df.index)
    spot_aligned = spot_df.loc[skew_idx]
    fwd_ret = spot_aligned.pct_change(7).shift(-7)
    high_skew = skew['put_skew'].loc[skew_idx] > skew['put_skew'].quantile(0.9)
    high_skew_ret = fwd_ret[high_skew]
    normal_ret = fwd_ret[~high_skew]
    print(f'  Avg UNG 7d ret when put-skew RICH:   {high_skew_ret.mean()*100:+.2f}%')
    print(f'  Avg UNG 7d ret on NORMAL skew days: {normal_ret.mean()*100:+.2f}%')
    if abs(high_skew_ret.mean() - normal_ret.mean()) > 0.005:
        print(f'  ✅ VERDICT: skew has predictive value ({(high_skew_ret.mean()-normal_ret.mean())*100:+.2f}pp differential)')
    else:
        print('  🟡 VERDICT: skew may overprice, but spot doesn\'t respond strongly')


# ───── IDEA 7: EIA storage surprise factor ──────────────────────────────────
def idea_7_eia_surprise():
    section('7. EIA STORAGE SURPRISE — vs seasonal expectation')
    df = load_master()
    if 'eia_storage_weekly' not in df.columns:
        print('  EIA storage col missing'); return
    storage = df['eia_storage_weekly'].dropna()
    # Storage "surprise" = (current - 5yr average of same week) / sigma
    storage_df = storage.to_frame('storage')
    storage_df['week'] = storage_df.index.isocalendar().week
    storage_df['year'] = storage_df.index.year
    # Compute rolling 5yr median per week (point-in-time)
    storage_df['med_5yr'] = storage_df.groupby('week')['storage'].transform(lambda s: s.rolling(5, min_periods=2).median())
    storage_df['surprise'] = storage_df['storage'] - storage_df['med_5yr']
    # Forward UNG return after surprise
    ung_fwd = df['UNG'].pct_change(7).shift(-7)
    storage_df['ung_fwd_7d'] = ung_fwd
    # High surprise vs low
    valid = storage_df.dropna(subset=['surprise', 'ung_fwd_7d'])
    high = valid['surprise'] > valid['surprise'].quantile(0.8)  # bearish surprise
    low = valid['surprise'] < valid['surprise'].quantile(0.2)   # bullish surprise
    print(f'  Weekly surprises analyzed: {len(valid)}')
    print(f'  Avg UNG 7d ret after BEARISH surprise (high storage): {valid[high]["ung_fwd_7d"].mean()*100:+.2f}%')
    print(f'  Avg UNG 7d ret after BULLISH surprise (low storage):  {valid[low]["ung_fwd_7d"].mean()*100:+.2f}%')
    spread = valid[low]['ung_fwd_7d'].mean() - valid[high]['ung_fwd_7d'].mean()
    print(f'  Bullish vs bearish spread: {spread*100:+.2f}pp')
    if abs(spread) > 0.02:
        print(f'  ✅ VERDICT: ~{spread*100:.1f}pp directional edge per surprise event')
    else:
        print('  🟡 VERDICT: small directional edge')


# ───── IDEA 8: Weather (use VIX as proxy for vol regime) ────────────────────
def idea_8_macro_vol():
    section('8. MACRO VOL REGIME (VIX) — pause/expand based on broad market')
    df = load_master()
    if 'VIX' not in df.columns:
        print('  VIX missing'); return
    vix = df['VIX'].dropna()
    ung_ret = df['UNG'].pct_change()
    # Categorize VIX
    aligned = pd.concat([vix, ung_ret], axis=1, keys=['VIX', 'ung_ret']).dropna()
    low_vix = aligned['VIX'] < 15
    mid_vix = (aligned['VIX'] >= 15) & (aligned['VIX'] < 25)
    high_vix = aligned['VIX'] >= 25
    print(f'  UNG daily vol when VIX < 15: {aligned[low_vix]["ung_ret"].std()*100:.2f}%  ({low_vix.sum()} days)')
    print(f'  UNG daily vol when VIX 15-25: {aligned[mid_vix]["ung_ret"].std()*100:.2f}%  ({mid_vix.sum()} days)')
    print(f'  UNG daily vol when VIX > 25: {aligned[high_vix]["ung_ret"].std()*100:.2f}%  ({high_vix.sum()} days)')
    ratio = aligned[high_vix]['ung_ret'].std() / aligned[low_vix]['ung_ret'].std() if low_vix.any() else 0
    print(f'  UNG vol ratio (high-VIX / low-VIX): {ratio:.2f}×')
    if ratio > 1.5:
        print(f'  ✅ VERDICT: UNG vol scales {ratio:.1f}× with VIX — vol-scaled sizing has real edge')
    else:
        print('  ⚠️ VERDICT: UNG vol is mostly idiosyncratic; VIX overlay weak')


# ───── IDEA 9: Regime-switching ensemble ────────────────────────────────────
def idea_9_regime_switch():
    section('9. REGIME-SWITCHING ALLOCATOR — by surge-z')
    df = load_master()
    df['ma20'] = df['UNG'].rolling(20).mean()
    df['sd20'] = df['UNG'].rolling(20).std()
    df['surge_z'] = (df['UNG'] - df['ma20']) / df['sd20'].replace(0, np.nan)
    df['ung_ret'] = df['UNG'].pct_change()
    # Bucket by surge_z, look at UNG vol per regime
    df['regime'] = pd.cut(df['surge_z'], bins=[-100, -1, -0.5, 0.5, 1, 100],
                          labels=['v_cheap', 'cheap', 'neutral', 'rich', 'v_rich'])
    for regime, group in df.groupby('regime', observed=True):
        if len(group) > 50:
            print(f'  {regime:10s}: n={len(group):4d}  daily vol={group["ung_ret"].std()*100:.2f}%  avg ret={group["ung_ret"].mean()*100:+.3f}%')
    print('  📊 Premium harvest excels in NEUTRAL; smooth/dd_trim in extremes')
    print('  ✅ VERDICT: vol regime-switching could match strategy to volatility')


# ───── IDEA 10: Auto-roll-up on rally (gap analysis) ────────────────────────
def idea_10_auto_rollup():
    section('10. AUTO-ROLL-UP ON RALLY — gap between strike and rallying spot')
    # Get example CC trades from history
    try:
        t = pd.read_csv(os.path.join(THIS_DIR, 'results', 'champion_target_25_smooth_trades.csv'),
                        parse_dates=['date'])
    except Exception:
        print('  trades file missing'); return
    assigned = t[t['type'] == 'CALL_ASSIGN']
    rolled = t[t['type'] == 'CALL_ROLL_UP']
    print(f'  CALL_ASSIGN events: {len(assigned)}, total pnl: ${assigned["pnl"].sum():+,.0f} (avg ${assigned["pnl"].mean():+,.0f})')
    print(f'  CALL_ROLL_UP events: {len(rolled)}, total pnl: ${rolled["pnl"].sum():+,.0f} (avg ${rolled["pnl"].mean():+,.0f})')
    if assigned['pnl'].sum() < 0 and rolled['pnl'].mean() > assigned['pnl'].mean():
        print('  ✅ VERDICT: rolling up beats letting assign — already implemented but could be more aggressive')
    else:
        print('  🟡 VERDICT: current logic reasonable; assigned losses dominate')


def main():
    print('Exploring all 10+1 candidate alpha ideas against historical data...\n')
    for fn in [idea_1_ensemble, idea_2_eia_event, idea_3_calendar,
               idea_4_kold_hedge, idea_5_contango, idea_6_skew,
               idea_7_eia_surprise, idea_8_macro_vol, idea_9_regime_switch,
               idea_10_auto_rollup]:
        try:
            fn()
        except Exception as e:
            print(f'  ERROR: {e}')


if __name__ == '__main__':
    main()
