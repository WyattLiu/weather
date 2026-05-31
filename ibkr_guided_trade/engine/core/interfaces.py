"""Engine contracts: INPUTS (data sources) → ENGINE → OUTPUTS (trades).

The engine knows nothing about HOW data is fetched or HOW trades are executed.
It just consumes typed inputs and produces typed outputs.

Live: adapters wrap WS API to produce these inputs
Backtest: adapters wrap historical CSVs + replay logic to produce same inputs
Both: same engine code path
"""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Protocol


# ══════════════════════════════════════════════════════════
# INPUT TYPES (what feeds INTO the engine)
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OptionQuote:
    """A single real option chain quote (NOT computed from BS)."""
    expiry: str               # 'YYYY-MM-DD'
    strike: float
    right: str                # 'C' or 'P'
    bid: float
    ask: float
    last: float
    open_interest: int = 0
    volume: int = 0
    iv: float = 0.0           # implied vol from market quote


@dataclass(frozen=True)
class MarketData:
    """Current market snapshot for one symbol."""
    symbol: str               # 'UNG', 'KOLD', 'NG', etc.
    bid: float
    ask: float
    last: float
    timestamp: Optional[str] = None


@dataclass
class OptionChain:
    """Full option chain for a symbol, organized by expiry → strike → quote."""
    symbol: str
    quotes: List[OptionQuote] = field(default_factory=list)

    def for_expiry(self, expiry: str) -> List[OptionQuote]:
        return [q for q in self.quotes if q.expiry == expiry]

    def nearest_strike(self, target: float, right: str, expiry: str) -> Optional[OptionQuote]:
        candidates = [q for q in self.quotes if q.expiry == expiry and q.right == right]
        if not candidates:
            return None
        return min(candidates, key=lambda q: abs(q.strike - target))


@dataclass(frozen=True)
class FundamentalSnapshot:
    """NG fundamentals at a point in time."""
    as_of: date
    storage_bcf: Optional[float] = None
    days_supply: Optional[float] = None
    consumption_bcf: Optional[float] = None
    production_bcf: Optional[float] = None
    lng_exports_bcf: Optional[float] = None
    storage_z: Optional[float] = None        # pre-computed if available
    composite_z: Optional[float] = None      # full multi-factor z from model
    regime: Optional[str] = None


@dataclass
class Position:
    """A held position — share or option."""
    kind: str                 # 'SHARES' | 'OPTION' | 'BOXX' | 'KOLD_SHARES'
    symbol: str
    qty: int                  # positive = long, negative = short
    avg_cost: float
    # Option-only fields:
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None


@dataclass
class Portfolio:
    """Holdings + cash."""
    as_of: date
    cash_usd: float
    buying_power_usd: float
    nlv_usd: float
    positions: List[Position] = field(default_factory=list)

    def shares(self, symbol: str = 'UNG') -> int:
        for p in self.positions:
            if p.kind == 'SHARES' and p.symbol == symbol:
                return p.qty
        return 0

    def options(self, symbol: str = 'UNG') -> List[Position]:
        return [p for p in self.positions if p.kind == 'OPTION' and p.symbol == symbol]


@dataclass
class EngineInputs:
    """The complete input bundle the engine consumes per decision cycle."""
    as_of: date
    portfolio: Portfolio
    ung_market: MarketData
    ung_chain: OptionChain
    fundamentals: FundamentalSnapshot
    # Optional inputs (engine adapts if missing)
    kold_market: Optional[MarketData] = None
    kold_chain: Optional[OptionChain] = None
    boil_market: Optional[MarketData] = None
    ng_market: Optional[MarketData] = None
    vix: Optional[float] = None


# ══════════════════════════════════════════════════════════
# OUTPUT TYPES (what the engine produces)
# ══════════════════════════════════════════════════════════

@dataclass
class TradeRecommendation:
    """A single recommended trade with explicit price + sizing."""
    action: str               # human-readable: "Sell 5x 6/26 $11.5P @ $0.55"
    side: str                 # 'BUY' | 'SELL'
    instrument: str           # 'UNG', 'UNG OPT', 'BOXX', 'KOLD'
    qty: int
    limit_price: float        # exact price (NOT auto-adjusted)
    # Option-specific:
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    open_close: str = 'OPEN'  # OPEN / CLOSE
    # Decision context:
    expected_cash_delta: float = 0.0
    regime: Optional[str] = None
    reasoning: str = ''
    confidence: float = 0.0   # 0-1 score from beam evaluator
    quality_delta: float = 0.0


@dataclass
class EngineOutputs:
    """Full output bundle from one decision cycle."""
    as_of: date
    regime: str
    composite_z: float
    recommendations: List[TradeRecommendation] = field(default_factory=list)
    forward_projection: Optional[dict] = None
    diagnostics: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════
# DATA SOURCE PROTOCOLS (adapters implement these)
# ══════════════════════════════════════════════════════════

class DataSource(Protocol):
    """Adapters (live WS, historical CSV) must implement this."""

    def get_inputs(self, as_of: date) -> EngineInputs:
        """Return the complete EngineInputs for the given date.
        Live: as_of is usually today; backtest: any historical date."""
        ...


class TradeExecutor(Protocol):
    """How trades are realized."""

    def execute(self, recommendation: TradeRecommendation, as_of: date) -> dict:
        """Realize a trade. Returns fill info / new state delta."""
        ...
