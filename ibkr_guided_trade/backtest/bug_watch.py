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
PERFECT_WIN_AVG_THRESH = 500.0   # ...but only flag if avg/fire is large.
# Conditional-outcome kernels fire ONLY when the outcome is positive (e.g., TP
# only fires when premium drops past threshold; EXPIRE_OTM only fires when OTM
# = full premium kept). 100% win is mathematically forced by the trigger, not
# evidence of pricing bugs. Skip the "too perfect" check for these.
CONDITIONAL_OUTCOME_KERNELS = {
    'PUT_TP', 'CALL_TP',
    'PUT_EXPIRE_OTM', 'CALL_EXPIRE_OTM',
    'LONG_PUT_EXPIRE',  # records realized payout, only "wins" if ITM at expiry
    'ELEVATOR_CLOSE',   # only fires when peak + low extrinsic — winning is the trigger
}
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

    # COVERED CALL ONLY policy: short_calls * 100 must never exceed shares.
    # Any violation = naked call sold (forbidden by user rule, see
    # [[feedback_covered_calls_only]])
    if 'shares' in hist.columns and 'short_calls' in hist.columns:
        gap = hist['short_calls'] * 100 - hist['shares']
        violations = hist[gap > 0]
        if not violations.empty:
            worst_idx = int(gap.idxmax())
            sc = int(violations.iloc[worst_idx]['short_calls']) if worst_idx < len(violations) else 0
            sh = int(violations.iloc[worst_idx]['shares']) if worst_idx < len(violations) else 0
            d = str(violations.iloc[worst_idx]['date']) if worst_idx < len(violations) else '?'
            report.add('ERROR', name, 'naked_calls_sold',
                      f'short_calls exceeded shares on {len(violations)} days. '
                      f'Worst: {sc} contracts vs {sh} shares on {d}')


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

    # Per-kernel 100% win rate w/ many fires AND large avg = real pricing concern.
    # 100% win at small avg is mathematically expected for conditional triggers
    # (TP fires only when premium dropped past threshold), not a bug.
    if 'pnl' in trades.columns:
        for kernel in counts.index:
            if kernel in CONDITIONAL_OUTCOME_KERNELS:
                continue
            sub = trades[trades['type'] == kernel]
            if len(sub) < PERFECT_WIN_FIRES_MIN:
                continue
            wins = (sub['pnl'] > 0).sum()
            if wins == len(sub):
                avg = float(sub['pnl'].mean())
                if avg > PERFECT_WIN_AVG_THRESH:
                    report.add('WARN', name, f'{kernel}_too_perfect',
                              f'{kernel} fired {len(sub)} times, 100% wins, '
                              f'${avg:.0f}/fire avg — suspiciously large; '
                              f'verify BS pricing not synthesizing free lunch')


def check_dataset_signals(report: BugReport):
    """Sanity check the master dataset's derived signals.

    Per [[feedback_rolling_window_calendar_vs_trading_days]]: trend
    signals stuck at one value for >95% of days are almost certainly
    rolling-window starvation bugs (pandas .rolling() on calendar-padded
    data with NaN weekends).
    """
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    try:
        from replay_engine import precompute_factor_z  # type: ignore
    except Exception as e:
        report.add('WARN', 'dataset', 'precompute_unavailable', str(e))
        return

    cache_path = os.path.join(_here, 'cache', 'master_dataset.csv')
    if not os.path.exists(cache_path):
        return
    df = pd.read_csv(cache_path, parse_dates=['Date'], index_col=0)
    df = precompute_factor_z(df).dropna(subset=['UNG'])

    # Signals that should vary day-to-day in a long sample
    bool_signals = ['ung_uptrend', 'ung_downtrend', 'ung_at_20d_low']
    for sig in bool_signals:
        if sig not in df.columns:
            continue
        true_pct = df[sig].mean() * 100  # fraction True
        # Any boolean stuck at <1% OR >99% is suspicious
        if true_pct < 1.0:
            report.add('WARN', 'dataset', f'{sig}_starved',
                      f'{sig} True only {true_pct:.1f}% of days — '
                      'rolling-window starvation likely')
        elif true_pct > 99.0:
            report.add('WARN', 'dataset', f'{sig}_saturated',
                      f'{sig} True {true_pct:.1f}% of days — '
                      'signal not discriminating')

    # Continuous signals that should not be 100% NaN
    cont_signals = ['ung_50d_ma', 'ung_200d_ma', 'storage_surprise_z',
                    'storage_z', 'days_supply_z']
    for sig in cont_signals:
        if sig not in df.columns:
            continue
        nan_pct = df[sig].isna().mean() * 100
        if nan_pct > 50.0:
            report.add('WARN', 'dataset', f'{sig}_mostly_nan',
                      f'{sig} is NaN {nan_pct:.0f}% of days — '
                      'rolling-window starvation likely')


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

    # Preventive: dataset-level signal sanity (catches calibration bugs
    # that wouldn't show up in per-strategy outputs)
    check_dataset_signals(report)

    return report


if __name__ == '__main__':
    report = run_checks()
    report.print()
    sys.exit(1 if report.has_errors else 0)
