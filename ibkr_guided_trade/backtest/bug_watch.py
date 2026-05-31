"""Sanity checks on backtest output.

User principle: losing lots of money can be a bug; making lots of money
can also be a bug. Either outlier might mean:
  - Option pricing miscalibrated (free lunch synthesized by BS)
  - Cash going negative without margin enforcement
  - Position counts blowing up
  - NaN/inf propagation
  - Same kernel firing in a loop

Run after every backtest; emit warnings (exit code 1) if anything trips.
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# Thresholds — these are heuristic, not laws of physics
SUSPICIOUS_RETURN_HIGH = 500.0   # > +500% in 5yr = check pricing
SUSPICIOUS_RETURN_LOW  = -100.0  # < -100% = cash blew up
MAX_KERNEL_FIRES       = 3000    # > 3000 fires of one kernel = loop?
PERFECT_WIN_FIRES_MIN  = 50      # 100% win after >=50 fires = synthetic pricing?
MAX_NAV_NEG_PCT        = 0.5     # NAV < -50% of initial = blown up
CASH_NEG_THRESHOLD     = -100000 # Cash more negative than -$100K = margin abuse


class BugReport:
    def __init__(self):
        self.findings = []  # list of dicts

    def add(self, severity: str, strategy: str, check: str, detail: str):
        self.findings.append({
            'severity': severity,
            'strategy': strategy,
            'check': check,
            'detail': detail,
        })

    @property
    def has_errors(self) -> bool:
        return any(f['severity'] == 'ERROR' for f in self.findings)

    def print(self):
        if not self.findings:
            print("[bug_watch] All checks passed.")
            return
        print(f"[bug_watch] {len(self.findings)} findings:")
        for f in self.findings:
            tag = '🚨' if f['severity'] == 'ERROR' else '⚠️ '
            print(f"  {tag} [{f['severity']}] {f['strategy']}.{f['check']}: {f['detail']}")


def check_summary(report: BugReport, summary: dict):
    """Sanity check the summary.json output."""
    for name, r in summary.items():
        if not isinstance(r, dict):
            continue
        if 'return_pct' not in r:
            continue
        ret = r.get('return_pct', 0)
        sharpe = r.get('sharpe', 0)
        final = r.get('final', 0)
        max_dd = r.get('max_dd_pct', 0)

        # NaN / inf checks
        for field, val in [('return_pct', ret), ('sharpe', sharpe),
                           ('final', final), ('max_dd_pct', max_dd)]:
            if val != val:  # NaN check
                report.add('ERROR', name, f'nan_{field}',
                          f'{field} is NaN — likely division by zero or bad data')
            elif val in (float('inf'), float('-inf')):
                report.add('ERROR', name, f'inf_{field}',
                          f'{field} is infinite')

        # Outlier returns
        if ret > SUSPICIOUS_RETURN_HIGH:
            report.add('WARN', name, 'too_good',
                      f'+{ret:.0f}% over 5yr — verify option pricing not synthesized free lunch')
        if ret < SUSPICIOUS_RETURN_LOW:
            report.add('ERROR', name, 'blown_up',
                      f'{ret:.0f}% — NAV went deeply negative (cash leverage bug?)')

        # Implausibly high Sharpe (UNG wheel can't realistically exceed ~2.0)
        if sharpe > 3.0:
            report.add('WARN', name, 'too_good_sharpe',
                      f'Sharpe {sharpe:.2f} unrealistically high — pricing or vol miscalibrated')


def check_history(report: BugReport, name: str, hist: pd.DataFrame):
    """Sanity check the per-strategy history."""
    if hist.empty:
        report.add('ERROR', name, 'empty_history', 'history dataframe empty')
        return

    initial_nav = float(hist.iloc[0]['nav'])

    # NaN in NAV
    nan_count = hist['nav'].isna().sum()
    if nan_count > 0:
        report.add('ERROR', name, 'nan_nav',
                  f'{nan_count} days with NaN NAV')

    # NAV blew up
    min_nav = float(hist['nav'].min())
    if min_nav < -MAX_NAV_NEG_PCT * initial_nav:
        report.add('ERROR', name, 'nav_blown_up',
                  f'min NAV ${min_nav:,.0f} < -{MAX_NAV_NEG_PCT*100:.0f}% of initial')

    # Cash deeply negative for sustained period
    if 'cash' in hist.columns:
        neg_days = (hist['cash'] < CASH_NEG_THRESHOLD).sum()
        if neg_days > 30:
            report.add('WARN', name, 'cash_neg_sustained',
                      f'cash < ${CASH_NEG_THRESHOLD:,} for {neg_days} days — margin abuse?')

    # Position count explosion
    for col in ('short_puts', 'short_calls'):
        if col not in hist.columns:
            continue
        max_pos = int(hist[col].max())
        if max_pos > 100:
            report.add('WARN', name, f'{col}_explosion',
                      f'max {col} = {max_pos} — loop bug? unrealistic concentration?')


def check_trades(report: BugReport, name: str, trades: pd.DataFrame):
    """Sanity check the trade log."""
    if trades.empty:
        return

    if 'type' not in trades.columns:
        return

    counts = trades['type'].value_counts()
    for kernel, n in counts.items():
        if n > MAX_KERNEL_FIRES:
            report.add('WARN', name, f'{kernel}_loop',
                      f'{kernel} fired {n} times — possible loop')

    # Per-kernel 100% win rate w/ many fires = pricing is too perfect
    if 'pnl' in trades.columns:
        for kernel in counts.index:
            sub = trades[trades['type'] == kernel]
            if len(sub) < PERFECT_WIN_FIRES_MIN:
                continue
            wins = (sub['pnl'] > 0).sum()
            if wins == len(sub):
                report.add('WARN', name, f'{kernel}_too_perfect',
                          f'{kernel} fired {len(sub)} times with 100% wins — '
                          f'BS-synthesized free lunch? real chains would have losses.')


def run_checks() -> BugReport:
    report = BugReport()

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    if not os.path.exists(summary_path):
        report.add('ERROR', '*', 'no_summary',
                  f'{summary_path} missing — backtest never ran')
        return report

    with open(summary_path) as f:
        summary = json.load(f)
    check_summary(report, summary)

    for name in summary:
        if not isinstance(summary.get(name), dict):
            continue
        hist_path = os.path.join(RESULTS_DIR, f'{name}_history.csv')
        trades_path = os.path.join(RESULTS_DIR, f'{name}_trades.csv')
        if os.path.exists(hist_path):
            hist = pd.read_csv(hist_path)
            check_history(report, name, hist)
        if os.path.exists(trades_path):
            trades = pd.read_csv(trades_path)
            check_trades(report, name, trades)

    return report


if __name__ == '__main__':
    report = run_checks()
    report.print()
    sys.exit(1 if report.has_errors else 0)
