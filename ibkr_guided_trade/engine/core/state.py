"""Portfolio state — the single representation used by both live and backtest.

A PortfolioState is a snapshot at a moment in time. All engine functions
take a State as input and return either a new State (after applying a trade)
or recommendations.
"""
from dataclasses import dataclass, field, replace
from datetime import date
from typing import List, Optional


@dataclass
class OptionPosition:
    """A single option holding."""
    expiry: str              # 'YYYY-MM-DD'
    strike: float
    right: str               # 'C' or 'P'
    qty: int                 # positive = long, negative = short
    avg_cost: float          # premium paid/collected per share

    def dte(self, today: date) -> int:
        from datetime import datetime
        try:
            exp = datetime.strptime(self.expiry, '%Y-%m-%d').date()
            return max(0, (exp - today).days)
        except Exception:
            return 0


@dataclass
class PortfolioState:
    """Complete portfolio snapshot. Same shape for live and backtest."""
    # As-of date
    today: date

    # Market data
    spot: float                          # UNG spot price
    iv: float                            # 30-day IV estimate
    ng_price: float = 0.0                # NG futures
    vix: float = 0.0
    kold_spot: float = 0.0

    # Holdings
    shares: int = 0
    options: List[OptionPosition] = field(default_factory=list)
    boxx_shares: int = 0
    kold_shares: int = 0

    # Cash & capacity
    cash: float = 0.0
    buying_power: float = 0.0
    nlv: float = 0.0

    # Fundamentals (for z-score)
    storage_bcf: Optional[float] = None
    days_supply: Optional[float] = None
    cl_price: Optional[float] = None
    ng_trend: Optional[float] = None     # vs MA200

    # Recent income tracking
    avg_weekly_theta: float = 0.0

    def with_shares(self, new_shares: int) -> 'PortfolioState':
        return replace(self, shares=new_shares)

    def with_cash(self, new_cash: float) -> 'PortfolioState':
        return replace(self, cash=new_cash)

    def add_option(self, opt: OptionPosition) -> 'PortfolioState':
        new_opts = list(self.options) + [opt]
        return replace(self, options=new_opts)

    def short_options(self):
        return [o for o in self.options if o.qty < 0]

    def short_calls(self):
        return [o for o in self.options if o.qty < 0 and o.right == 'C']

    def short_puts(self):
        return [o for o in self.options if o.qty < 0 and o.right == 'P']

    def long_options(self):
        return [o for o in self.options if o.qty > 0]
