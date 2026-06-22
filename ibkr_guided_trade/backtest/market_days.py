"""Trading-day-aware staleness: count NYSE sessions, not calendar days, so weekends and market
holidays (Juneteenth, Good Friday, Thanksgiving, …) never trip a "data stale" banner.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar, GoodFriday, AbstractHolidayCalendar


class _NYSECalendar(AbstractHolidayCalendar):
    # US federal holidays the NYSE observes (drop Columbus/Veterans — markets open) + Good Friday.
    rules = [r for r in USFederalHolidayCalendar.rules
             if r.name not in ("Columbus Day", "Veterans Day")] + [GoodFriday]


_HOLIDAYS = _NYSECalendar().holidays(start="2015-01-01", end="2035-12-31").values.astype("datetime64[D]")


def trading_days_stale(asof, today=None) -> int:
    """Completed NYSE trading sessions strictly AFTER `asof` and strictly BEFORE `today`.

    `today` is excluded because its EOD data isn't expected until after the close — so on a normal
    day (data = yesterday's close), or over a weekend/holiday gap, this returns 0 (current-as-possible).
    A genuinely lagging feed returns the number of trading sessions actually missed.
    """
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    a = np.datetime64((pd.Timestamp(asof).normalize() + pd.Timedelta(days=1)).date(), "D")
    b = np.datetime64(today.normalize().date(), "D")
    if b <= a:
        return 0
    return int(np.busday_count(a, b, holidays=_HOLIDAYS))
