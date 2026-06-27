"""Minute-fill frontier: re-rank candidate kernels under INTRADAY minute execution
(intraday_exec) — the only honest fill model now. Full-sample + sealed OOS."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z, CACHE_DIR
TEST_START='2024-01-02'
FRONTIER=[('kold15_ivrank_kbh (LIVE)','champion_kold15_ivrank_kbh'),
          ('kold15_ivrank','champion_kold15_ivrank'),
          ('Router-safe g11','g11_router_safe'),
          ('PutRatio-2x g11','g11_putratio_big'),
          ('ITM-put g11','g11_itmput_conv'),
          ('Gap-Wheel g14','g14_gap_wheel_real'),
          ('Gen-10 book55','g10_book55')]
MIN={'intraday_exec':True,'exec_window':15,'avoid_eia_print':True}
def metrics(nav):
    nav=nav.dropna()
    if len(nav)<30: return None
    r=nav.pct_change().dropna(); yrs=(nav.index[-1]-nav.index[0]).days/365.25
    ann=((nav.iloc[-1]/nav.iloc[0])**(1/yrs)-1)*100 if yrs>0 else 0
    sh=r.mean()/(r.std()+1e-12)*math.sqrt(252)
    mdd=((nav/nav.cummax()-1).min())*100
    return round(ann,1),round(sh,2),round(mdd,1)
df=pd.read_csv(os.path.join(CACHE_DIR,'master_dataset.csv'),index_col=0,parse_dates=True)
df=precompute_factor_z(df).dropna(subset=['UNG'])
dft=df.loc[TEST_START:]
rows=[]
for label,key in FRONTIER:
    if key not in STRATEGIES:
        print(f"SKIP {key}"); continue
    p={**STRATEGIES[key],**MIN}
    hf,_=run_strategy_simple(df,p,48000,6200); hf=hf.set_index(pd.to_datetime(hf['date']))
    ho,_=run_strategy_simple(dft,p,100000,0); ho=ho.set_index(pd.to_datetime(ho['date']))
    mf,mo=metrics(hf['nav']),metrics(ho['nav'])
    if not mf or not mo: continue
    rows.append({'label':label,'key':key,'full_ann':mf[0],'full_sharpe':mf[1],'full_mdd':mf[2],
                 'oos_ann':mo[0],'oos_sharpe':mo[1],'oos_mdd':mo[2]})
    print(f"{label:24} FULL {mf[0]:+6.1f}%/{mf[1]:.2f}/{mf[2]:.0f}  OOS {mo[0]:+6.1f}%/{mo[1]:.2f}/{mo[2]:.0f}",flush=True)
res=pd.DataFrame(rows).sort_values('oos_sharpe',ascending=False)
res.to_csv(os.path.join(os.path.dirname(__file__),'results','minute_frontier.csv'),index=False)
open(os.path.join(os.path.dirname(__file__),'log','minute_frontier.txt'),'w').write(
    "MINUTE-FILL frontier (sorted by OOS Sharpe):\n"+res.to_string(index=False))
print("\nBEST by OOS Sharpe:",res.iloc[0]['label'] if len(res) else "none")
