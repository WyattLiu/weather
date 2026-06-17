# Pre-minute backtests are OBSOLETE (cutover 2026-06-16)

All artifacts in `obsolete_preminute/` were produced under **optimistic / EOD fills**
(Black-Scholes model mid or single EOD bid/ask snapshot). They are **superseded** and
must not be cited.

The ONLY valid basis going forward is **real-fill, minute-traded, audit-verifiable**
execution: every fill priced from the intraday minute bid/ask path
(`market_scanner.ung_options_history`, 198M rows, RTH-only, two-sided), stamped with
exec_time / bid / ask / spread / source via `exec_fill()`. See
[[project_intraday_execution_audit]].

Canonical minute-fill results live directly in `results/` (regenerated under
`intraday_exec`). Kernel rankings: `results/minute_frontier.csv`.
