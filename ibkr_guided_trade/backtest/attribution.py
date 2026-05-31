"""Per-kernel P&L attribution.

Every leg gets a leg_id at open. Every close (TP, roll, expiry, assignment,
elevator) emits a P&L record tagged with open_kernel + close_kernel.

Aggregation produces: per-kernel fire count, total P&L, win rate,
P&L distribution, and contribution to total return.
"""
from collections import defaultdict
from typing import Iterable

import pandas as pd


def new_leg_id(counter: dict) -> int:
    counter['n'] = counter.get('n', 0) + 1
    return counter['n']


def attribute_trades(trades: Iterable[dict]) -> pd.DataFrame:
    """Aggregate trade records into per-kernel attribution table.

    Each trade record should include:
      - type:          kernel name (PUT_TP, PUT_ASSIGN, CALL_ROLL_UP, etc.)
      - pnl (optional): realized $ P&L for the action; if missing, inferred 0
      - date:          when it fired

    Returns DataFrame with columns:
      kernel, count, total_pnl, avg_pnl, win_count, win_rate, std_pnl
    """
    by_kernel: dict = defaultdict(list)
    for t in trades:
        k = t.get('type', 'UNKNOWN')
        pnl = t.get('pnl')
        if pnl is None:
            # Try to infer from common fields
            if 'locked_gain' in t:
                pnl = float(t['locked_gain'])
            else:
                pnl = 0.0
        by_kernel[k].append(float(pnl))

    rows = []
    for k, pnls in sorted(by_kernel.items()):
        s = pd.Series(pnls)
        rows.append({
            'kernel': k,
            'count': len(pnls),
            'total_pnl': float(s.sum()),
            'avg_pnl': float(s.mean()),
            'win_count': int((s > 0).sum()),
            'win_rate': float((s > 0).mean()) if len(s) else 0.0,
            'std_pnl': float(s.std()) if len(s) > 1 else 0.0,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('total_pnl', ascending=False).reset_index(drop=True)
    return df


def print_attribution(name: str, attr: pd.DataFrame, total_nav_delta: float):
    """Pretty-print attribution table."""
    print(f"\n=== {name} — Kernel Attribution ===")
    if attr.empty:
        print("(no trades)")
        return
    print(f"{'kernel':<24} {'n':>5} {'total $':>12} {'avg $':>9} "
          f"{'win%':>6} {'contrib%':>9}")
    print("-" * 70)
    for _, r in attr.iterrows():
        contrib = (r['total_pnl'] / total_nav_delta * 100) if total_nav_delta else 0
        print(f"{r['kernel']:<24} {int(r['count']):>5} ${r['total_pnl']:>10,.0f} "
              f"${r['avg_pnl']:>8,.0f} {r['win_rate']*100:>5.0f}% "
              f"{contrib:>+8.1f}%")
    total = attr['total_pnl'].sum()
    print(f"{'TOTAL attributed':<24} {'':>5} ${total:>10,.0f} {'':>9} {'':>6} "
          f"{(total/total_nav_delta*100 if total_nav_delta else 0):>+8.1f}%")
