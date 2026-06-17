import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, datetime as dt, replay_engine as R
df=pd.read_csv(os.path.join(R.CACHE_DIR,'master_dataset.csv'), parse_dates=[0], index_col=0)
df=R.precompute_factor_z(df).dropna(subset=['UNG'])
params={**R.STRATEGIES['champion_kold15_ivrank_kbh'],'intraday_exec':True,'exec_window':15,'avoid_eia_print':True}
h,t=R.run_strategy_simple(df.loc['2021-06-17':], params, 48000, 6200)
sites=['OPEN_PUT','OPEN_CC','OPEN_ITM_CC','PUT_TP','CALL_TP','PUT_ROLL_DOWN']
out=[f"trades total: {len(t)}  span {t['date'].min().date()} -> {t['date'].max().date()}",
     f"{'site':14} {'n':>5} {'audit':>6} {'intraday':>9} {'eod':>5} {'model':>6} {'RTH%':>6} {'min%':>6}"]
ta=tr=tm=tn=0
for s in sites:
    sub=t[t['type']==s]
    if not len(sub): continue
    aud=sub['fill_source'].notna()
    et=pd.to_datetime(sub.loc[aud,'exec_time'], errors='coerce').dropna()
    rth=((et.dt.time>=dt.time(9,30))&(et.dt.time<=dt.time(16,0))).sum()
    mn=(et.dt.minute!=30).sum()
    vc=sub.loc[aud,'fill_source'].value_counts().to_dict()
    out.append(f"{s:14} {len(sub):5} {aud.sum():6} {vc.get('intraday',0):9} {vc.get('eod_real',0):5} {vc.get('model',0):6} {100*rth/max(1,len(et)):5.1f} {100*mn/max(1,len(et)):5.1f}")
    ta+=aud.sum(); tr+=rth; tm+=mn; tn+=len(et)
out.append(f"\nTOTAL audited:{ta}  RTH:{100*tr/max(1,tn):.2f}%  minute-granular:{100*tm/max(1,tn):.1f}%")
allet=pd.to_datetime(t['exec_time'], errors='coerce').dropna()
oob=allet[(allet.dt.time<dt.time(9,30))|(allet.dt.time>dt.time(16,0))]
out.append(f"out-of-RTH fills: {len(oob)} (must be 0)")
out.append("sample exec minutes: "+", ".join(sorted(allet.dt.strftime('%H:%M').unique())[:15]))
open('backtest/log/audit_validation.txt','w').write("\n".join(out))
print("DONE")
