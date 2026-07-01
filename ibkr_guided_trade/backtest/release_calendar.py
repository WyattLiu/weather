"""Release calendar — the EXACT public moment of each data series' value (roadmap FiA).

The minute-reactive, no-leak backtest must never let a decision at time T use a value that wasn't public
by T. The single biggest trap is an EIA print: at minute granularity the number must appear at its exact
release instant, not at the report-period date or the day's open. This module encodes those release rules
as pure, tz-aware functions the reactive engine gates on (FiC) and the event-exact test asserts (FiB).

Rules (all US/Eastern):
  - Weekly NG storage: report covers the week ending a Friday; RELEASED the following Thursday 10:30 ET.
  - EIA-914 monthly (production/consumption/LNG): reference month M; RELEASED ~end of month M+2. We use the
    last business day of M+2 at 12:00 ET — an approximation that ERRS LATE (conservative = never leaks; a
    slightly-too-late availability only costs timeliness, never correctness).
  - Prices / VIX: available at the bar timestamp itself.
"""
import datetime as dt

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('America/New_York')
except Exception:                       # pragma: no cover
    ET = dt.timezone(dt.timedelta(hours=-5))


def storage_release_ts(week_ending_friday):
    """EIA weekly NG storage for the week ending `week_ending_friday` → released the FOLLOWING Thursday
    10:30 ET. (Friday weekday=4 → +6 days = next Thursday.)"""
    d = _as_date(week_ending_friday)
    days = (3 - d.weekday()) % 7          # Thursday = 3
    if days == 0:
        days = 7
    r = d + dt.timedelta(days=days)
    return dt.datetime(r.year, r.month, r.day, 10, 30, tzinfo=ET)


def monthly_release_ts(reference_month_any_day):
    """EIA-914 monthly for the reference month containing `reference_month_any_day` → released ~end of
    month M+2 (last business day, 12:00 ET). Conservative (errs late)."""
    d = _as_date(reference_month_any_day)
    # first day of month M+2
    m = d.month - 1 + 2
    y = d.year + m // 12
    m = m % 12 + 1
    # last calendar day of that month
    if m == 12:
        last = dt.date(y, 12, 31)
    else:
        last = dt.date(y, m + 1, 1) - dt.timedelta(days=1)
    while last.weekday() >= 5:            # back up to a business day
        last -= dt.timedelta(days=1)
    return dt.datetime(last.year, last.month, last.day, 12, 0, tzinfo=ET)


def price_release_ts(bar_datetime):
    """Prices/VIX are public at the bar itself."""
    b = bar_datetime
    if isinstance(b, dt.datetime):
        return b if b.tzinfo else b.replace(tzinfo=ET)
    d = _as_date(b)                        # a date → assume the 16:00 ET close of that day
    return dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)


_DISPATCH = {
    'eia_storage_weekly': storage_release_ts, 'days_supply': storage_release_ts,
    'eia_production': monthly_release_ts, 'eia_consumption': monthly_release_ts,
    'eia_lng_exports': monthly_release_ts, 'eia_pipe_exports': monthly_release_ts,
}


def release_ts(series, period):
    """Exact public datetime (ET, tz-aware) of `series`'s value for report period `period`.
    Unknown/price series → available at the bar (price_release_ts)."""
    return _DISPATCH.get(series, price_release_ts)(period)


def _as_date(x):
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    import pandas as pd
    return pd.Timestamp(x).date()
