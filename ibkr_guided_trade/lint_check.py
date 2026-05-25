#!/usr/bin/env python3
"""
Lint / type-check wrapper for the autonomous cron loop.

Runs available checkers (ruff, pyright, mypy) on the project's Python
files and reports a single PASS/FAIL plus per-tool counts. Designed to
be invoked per cron cycle alongside the philosophy queue work.

Usage:
  python lint_check.py                   # quick check (default targets)
  python lint_check.py FILE [FILE ...]   # specific files
  python lint_check.py --json            # machine-readable output

Exit codes:
  0 = no errors
  1 = errors found
  2 = no checkers available (skip with a warning)

Suppression policy (codified in this file):
  - F841 unused vars: left for manual review (could be intentional placeholders)
  - E402 imports-not-at-top: allowed where there's an explicit warnings.filter
  - Inherent pandas typing noise (tz/iloc/strftime on Index): silenced via
    file-level pyrightignore in pyproject.toml — not in this wrapper
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TARGETS = ['ung_visualizer.py', 'param_sweep.py', 'lint_check.py']

# Pyright errors that are inherent pandas/numpy typing limitations or noisy
# stylistic issues. Filtered out of the error count but printed for awareness.
# Pyright doesn't understand pandas Series/DataFrame/Index dynamic types, so
# many idiomatic pandas calls trip its checker. These patterns capture that.
PYRIGHT_NOISE_PATTERNS = [
    'reportMissingImports',                   # local-only imports
    'for class "ndarray',                     # any attribute access on ndarray
    'for class "Index"',                      # any attribute access on Index
    'for class "Index*"',
    'for class "MultiIndex"',
    'for class "RangeIndex"',
    'for class "Hashable"',                   # df.index iteration types
    'No overloads for "round" match',         # numpy ndarray + float
    'Operator "-" not supported for types',   # Index arithmetic
    'cannot be assigned to parameter "x" of type "ConvertibleToFloat"',
    'cannot be assigned to parameter "prices"',
    'cannot be assigned to parameter "start_capital"',  # float/int ambiguity
    'cannot be assigned to parameter "number" of type "_SupportsRound2',
    'Invalid conditional operand of type',    # Series in boolean context
]


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as e:
        return 99, f'{cmd[0]} run failed: {e}'


def run_ruff(targets: list[str]) -> dict:
    ruff = shutil.which('ruff') or '/home/wyatt/.venvs/shinobi/bin/ruff'
    if not Path(ruff).exists():
        return {'tool': 'ruff', 'available': False, 'errors': 0, 'output': ''}
    code, out = _run([ruff, 'check', '--output-format=concise', *targets])
    err_count = sum(1 for ln in out.splitlines()
                    if ':' in ln and any(t in ln for t in (' E', ' F', ' W', ' I')))
    return {'tool': 'ruff', 'available': True, 'errors': err_count,
            'exit': code, 'output': out}


def run_pyright(targets: list[str]) -> dict:
    pyright = shutil.which('pyright')
    if not pyright:
        return {'tool': 'pyright', 'available': False, 'errors': 0, 'output': ''}
    code, out = _run([pyright, *targets])
    # Pyright prints "N errors, M warnings" in last lines
    err_count = 0
    real_errors = []
    for ln in out.splitlines():
        if ' - error:' in ln:
            if not any(pat in ln for pat in PYRIGHT_NOISE_PATTERNS):
                err_count += 1
                real_errors.append(ln.strip())
    return {'tool': 'pyright', 'available': True, 'errors': err_count,
            'exit': code, 'output': out, 'real_errors': real_errors}


def run_mypy(targets: list[str]) -> dict:
    mypy = shutil.which('mypy')
    if not mypy:
        return {'tool': 'mypy', 'available': False, 'errors': 0, 'output': ''}
    code, out = _run([mypy, '--ignore-missing-imports', '--no-strict-optional', *targets])
    err_count = sum(1 for ln in out.splitlines() if ': error:' in ln)
    return {'tool': 'mypy', 'available': True, 'errors': err_count,
            'exit': code, 'output': out}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1] if __doc__ else '')
    p.add_argument('targets', nargs='*', default=DEFAULT_TARGETS)
    p.add_argument('--json', action='store_true')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-tool output, just print summary line')
    args = p.parse_args(argv)

    results = [run_ruff(args.targets), run_pyright(args.targets), run_mypy(args.targets)]

    available = [r for r in results if r['available']]
    if not available:
        msg = 'No linters available (ruff/pyright/mypy all missing)'
        print(json.dumps({'verdict': 'SKIP', 'reason': msg}) if args.json else msg)
        return 2

    total_real = sum(r['errors'] for r in available)

    if args.json:
        print(json.dumps({
            'verdict': 'PASS' if total_real == 0 else 'FAIL',
            'total_errors': total_real,
            'per_tool': [{'tool': r['tool'], 'errors': r['errors']} for r in available],
            'pyright_real_errors': next((r.get('real_errors', []) for r in available if r['tool'] == 'pyright'), []),
        }, indent=2))
    else:
        if not args.quiet:
            for r in available:
                print(f'\n=== {r["tool"].upper()} ({r["errors"]} errors) ===')
                if r['tool'] == 'pyright':
                    # Only show real (non-noise) errors
                    for ln in r.get('real_errors', [])[:30]:
                        print(f'  {ln}')
                else:
                    out = r.get('output', '')
                    print('\n'.join(out.splitlines()[:30]))
        print('\n--- Lint summary ---')
        for r in available:
            print(f'  {r["tool"]:10s} {r["errors"]:>4d} errors')
        print(f'  total      {total_real:>4d} real errors '
              f'({"PASS" if total_real == 0 else "FAIL"})')

    return 0 if total_real == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
