"""#1 refined-signal discovery + #4 high-VIX continuation discovery, from spy_allday.csv (instant).
Tests which gate-stacks lift the low-VIX straddle, and whether ANY high-VIX subset is profitable.

  venv/bin/python research/spy_vol/spy_signal_stacks.py
"""
import os
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(THIS, 'spy_allday.csv'))


def stat(sub):
    if not len(sub):
        return "n=0"
    r = sub['ret']
    return f"n={len(sub):>4}  avg {r.mean():>+7.1%}  win {(r>0).mean()*100:>3.0f}%  big(>20%) {(r>0.20).mean()*100:>3.0f}%"


print("=== #1 REFINED SIGNAL — stacking gates onto VIX<=16 ===")
lv = df[df['vix'] <= 16]
print(f"  VIX<=16 (base)                 : {stat(lv)}")
print(f"  +near-highs (dist_high>-0.05)  : {stat(lv[lv['dist_high'] > -0.05])}")
print(f"  +low-rv (rv20<0.13)            : {stat(lv[lv['rv20'] < 0.13])}")
print(f"  +flat-skew (skew<0.04)         : {stat(lv[lv['skew'] < 0.04])}")
print(f"  +not-cheap (iv_rv>=0)          : {stat(lv[lv['iv_rv'] >= 0])}")
print(f"  +stable VIX (vix_chg5<1)       : {stat(lv[lv['vix_chg5'] < 1])}")
stk = lv[(lv['dist_high'] > -0.05) & (lv['rv20'] < 0.13) & (lv['skew'] < 0.04)]
print(f"  STACK near-highs+low-rv+flat   : {stat(stk)}")
stk2 = lv[(lv['dist_high'] > -0.05) & (lv['skew'] < 0.04)]
print(f"  STACK near-highs+flat          : {stat(stk2)}")
stk3 = lv[(lv['dist_high'] > -0.05) & (lv['rv20'] < 0.13)]
print(f"  STACK near-highs+low-rv        : {stat(stk3)}")

print("\n=== #4 HIGH-VIX continuation — is ANY high-VIX subset profitable? ===")
hv = df[df['vix'] > 16]
print(f"  VIX>16 (all)                   : {stat(hv)}")
print(f"  VIX>16 & rising (vix_chg5>2)   : {stat(hv[hv['vix_chg5'] > 2])}")
print(f"  VIX>16 & falling-back(chg5<-2) : {stat(hv[hv['vix_chg5'] < -2])}")
print(f"  VIX>16 & deep selloff(dist<-.1): {stat(hv[hv['dist_high'] < -0.10])}")
print(f"  VIX>16 & flat-skew(<0.04)      : {stat(hv[hv['skew'] < 0.04])}")
print(f"  VIX>16 & steep-skew(>0.06)     : {stat(hv[hv['skew'] > 0.06])}")
print(f"  VIX 16-22 (moderate)           : {stat(hv[(hv['vix'] > 16) & (hv['vix'] <= 22)])}")
print(f"  VIX>22 (high)                  : {stat(hv[hv['vix'] > 22])}")
print(f"  VIX>16 & near-highs(dist>-.03) : {stat(hv[hv['dist_high'] > -0.03])}")
print(f"  VIX>16 & calm-std(vix_std10<1) : {stat(hv[hv['vix_std10'] < 1])}")
print("\nDONE")
