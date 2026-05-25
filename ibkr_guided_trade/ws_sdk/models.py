"""Typed primitives for the ws_sdk package.

All of these types are frozen dataclasses / str-Enums so they are hashable
and safe to pass around. Each dataclass has a ``from_*`` classmethod that
accepts a raw GraphQL dict and normalises it into SDK types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, List, Optional


# ------------------------------------------------------------------ helpers
def _decimal(value: Any) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _optional_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle WS ISO format: "2026-04-07T14:11:45.895Z"
        try:
            s = value.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


# ================================================================== enums

class OrderStatus(str, Enum):
    """Normalised order status.

    Wealthsimple returns raw strings like ``"posted"``, ``"pending"``, etc.
    Use :meth:`from_raw` to convert. Unknown values fall back to PENDING
    rather than raising so the SDK remains forward-compatible.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    WORKING = "working"
    POSTED = "posted"           # WS uses "posted" to mean filled/closed
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> "OrderStatus":
        s = (raw or "").strip().lower()
        mapping = {
            "pending": cls.PENDING,
            "new": cls.PENDING,
            "submitted": cls.SUBMITTED,
            "accepted": cls.SUBMITTED,
            "working": cls.WORKING,
            "partially_filled": cls.WORKING,
            "posted": cls.POSTED,
            "filled": cls.POSTED,
            "completed": cls.POSTED,
            "cancelled": cls.CANCELLED,
            "canceled": cls.CANCELLED,
            "rejected": cls.REJECTED,
            "failed": cls.REJECTED,
            "expired": cls.EXPIRED,
        }
        return mapping.get(s, cls.PENDING)

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.POSTED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def is_filled(self) -> bool:
        return self == OrderStatus.POSTED


class OrderSide(str, Enum):
    BUY = "BUY_QUANTITY"
    SELL = "SELL_QUANTITY"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> Optional["OrderSide"]:
        if not raw:
            return None
        s = raw.strip().upper()
        if s in ("BUY", "BUY_QUANTITY"):
            return cls.BUY
        if s in ("SELL", "SELL_QUANTITY"):
            return cls.SELL
        return None


class OpenClose(str, Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"

    @classmethod
    def from_raw(cls, raw: Optional[str]) -> Optional["OpenClose"]:
        if not raw:
            return None
        s = raw.strip().upper()
        if s == "OPEN":
            return cls.OPEN
        if s == "CLOSE":
            return cls.CLOSE
        return None


class AccountType(str, Enum):
    """Classification of a Wealthsimple account.

    MARGIN is the only tradeable equity/options account; everything else
    (crypto, cash, credit card) is filtered out by the SDK by default.
    """

    MARGIN = "margin"
    CRYPTO = "crypto"
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    TFSA = "tfsa"
    RRSP = "rrsp"
    OTHER = "other"

    @classmethod
    def from_account_id(cls, acct_id: Optional[str]) -> "AccountType":
        if not acct_id:
            return cls.OTHER
        s = acct_id.lower()
        if "crypto" in s:
            return cls.CRYPTO
        if s.startswith("non-registered"):
            return cls.MARGIN
        if s.startswith("ca-cash"):
            return cls.CASH
        if s.startswith("ca-credit-card"):
            return cls.CREDIT_CARD
        if "tfsa" in s:
            return cls.TFSA
        if "rrsp" in s:
            return cls.RRSP
        return cls.OTHER


# ================================================================== Order

@dataclass(frozen=True)
class Order:
    """Normalised single-leg order snapshot.

    Populated from the ``soOrdersExtendedOrder`` GraphQL response. For
    multi-leg orders use :class:`MultilegOrder` which wraps a list of
    ``Order`` objects (one per leg).
    """

    external_id: str
    status: OrderStatus
    side: Optional[OrderSide]
    open_close: Optional[OpenClose]
    security_id: str
    submitted_quantity: Decimal
    filled_quantity: Decimal
    average_filled_price: Optional[Decimal]
    limit_price: Optional[Decimal]
    time_in_force: Optional[str]
    submitted_at: Optional[datetime]
    first_filled_at: Optional[datetime]
    last_filled_at: Optional[datetime]
    expired_at: Optional[datetime]
    rejection_cause: Optional[str]
    rejection_code: Optional[str]
    canonical_account_id: Optional[str]
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    # ------------- construction --------------------------------------
    @classmethod
    def from_extended(cls, data: dict, external_id: Optional[str] = None) -> "Order":
        """Build from a ``FetchSoOrdersExtendedOrder`` response node."""
        data = data or {}
        return cls(
            external_id=external_id or data.get("externalId") or "",
            status=OrderStatus.from_raw(data.get("status")),
            side=OrderSide.from_raw(data.get("orderType")),
            open_close=OpenClose.from_raw(data.get("openClose")),
            security_id=data.get("securityId") or "",
            submitted_quantity=_decimal(data.get("submittedQuantity")),
            filled_quantity=_decimal(data.get("filledQuantity")),
            average_filled_price=_optional_decimal(data.get("averageFilledPrice")),
            limit_price=_optional_decimal(data.get("limitPrice")),
            time_in_force=data.get("timeInForce"),
            submitted_at=_datetime(data.get("submittedAtUtc")),
            first_filled_at=_datetime(data.get("firstFilledAtUtc")),
            last_filled_at=_datetime(data.get("lastFilledAtUtc")),
            expired_at=_datetime(data.get("expiredAtUtc")),
            rejection_cause=data.get("rejectionCause"),
            rejection_code=data.get("rejectionCode"),
            canonical_account_id=data.get("canonicalAccountId") or data.get("accountId"),
            raw=dict(data),
        )

    @classmethod
    def from_multileg_leg(cls, leg: dict, batch_status: Optional[str] = None) -> "Order":
        """Build from a single leg of a ``FetchSoOrdersMultilegOrder`` response."""
        leg = leg or {}
        fill = leg.get("averageFillPrice") or {}
        return cls(
            external_id=leg.get("externalId") or leg.get("orderId") or "",
            status=OrderStatus.from_raw(leg.get("status") or batch_status),
            side=OrderSide.from_raw(leg.get("side")),
            open_close=OpenClose.from_raw(leg.get("openClose")),
            security_id=leg.get("securityId") or "",
            submitted_quantity=_decimal(leg.get("submittedQuantity")),
            filled_quantity=_decimal(leg.get("filledQuantity")),
            average_filled_price=_optional_decimal(fill.get("amount") if isinstance(fill, dict) else fill),
            limit_price=None,
            time_in_force=None,
            submitted_at=_datetime(leg.get("createdAtUtc")),
            first_filled_at=_datetime(leg.get("firstFilledAtUtc")),
            last_filled_at=_datetime(leg.get("lastFilledAtUtc")),
            expired_at=None,
            rejection_cause=None,
            rejection_code=None,
            canonical_account_id=None,
            raw=dict(leg),
        )

    # ------------- derived properties --------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_filled(self) -> bool:
        return self.status.is_filled

    @property
    def is_partially_filled(self) -> bool:
        return (
            self.status != OrderStatus.POSTED
            and self.filled_quantity > 0
            and self.filled_quantity < self.submitted_quantity
        )

    @property
    def remaining_quantity(self) -> Decimal:
        return max(Decimal("0"), self.submitted_quantity - self.filled_quantity)


# ================================================================== Multi-leg

@dataclass(frozen=True)
class MultilegOrder:
    """Aggregated view of a multi-leg order (straddle, spread, 4-leg roll)."""

    batch_id: str
    external_id: str
    status: OrderStatus
    strategy: Optional[str]
    limit_price: Optional[Decimal]
    time_in_force: Optional[str]
    submitted_at: Optional[datetime]
    updated_at: Optional[datetime]
    total_fee: Optional[Decimal]
    legs: List[Order]
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_response(cls, data: dict) -> "MultilegOrder":
        data = data or {}
        batch_status = data.get("status")
        legs = [Order.from_multileg_leg(leg, batch_status) for leg in (data.get("legs") or [])]
        return cls(
            batch_id=data.get("orderBatchId") or "",
            external_id=data.get("externalId") or "",
            status=OrderStatus.from_raw(batch_status),
            strategy=data.get("optionStrategy"),
            limit_price=_optional_decimal(data.get("limitPrice")),
            time_in_force=data.get("timeInForce"),
            submitted_at=_datetime(data.get("submittedAtUtc")),
            updated_at=_datetime(data.get("updatedAtUtc")),
            total_fee=_optional_decimal(data.get("totalFee")),
            legs=legs,
            raw=dict(data),
        )

    @property
    def is_terminal(self) -> bool:
        if self.status.is_terminal:
            return True
        # Also terminal if every leg is terminal
        return bool(self.legs) and all(leg.is_terminal for leg in self.legs)

    @property
    def is_filled(self) -> bool:
        if self.status.is_filled:
            return True
        return bool(self.legs) and all(leg.is_filled for leg in self.legs)


# ================================================================== Account

@dataclass(frozen=True)
class Account:
    id: str
    type: AccountType
    nickname: Optional[str] = None
    currency: Optional[str] = None
    status: Optional[str] = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_node(cls, node: dict) -> "Account":
        node = node or {}
        acct_id = node.get("id") or ""
        return cls(
            id=acct_id,
            type=AccountType.from_account_id(acct_id),
            nickname=node.get("nickname"),
            currency=node.get("currency"),
            status=node.get("status"),
            raw=dict(node),
        )

    @property
    def is_margin(self) -> bool:
        return self.type == AccountType.MARGIN


# ================================================================== Position

@dataclass(frozen=True)
class Position:
    security_id: str
    symbol: str
    quantity: Decimal
    average_price: Decimal
    market_value: Decimal
    book_value: Decimal
    unrealized_pnl: Decimal
    is_option: bool
    option_type: Optional[str]   # "CALL" / "PUT" / None
    strike: Optional[Decimal]
    expiry: Optional[str]
    underlying_symbol: Optional[str]
    accounts: List[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_position_v2(cls, node: dict) -> "Position":
        node = node or {}
        security = node.get("security") or {}
        stock = security.get("stock") or {}
        opt = security.get("optionDetails") or {}

        is_option = bool(opt)
        opt_type: Optional[str] = None
        strike: Optional[Decimal] = None
        expiry: Optional[str] = None
        underlying: Optional[str] = None
        symbol = stock.get("symbol") or ""

        if is_option:
            opt_type = opt.get("optionType")
            strike = _optional_decimal(opt.get("strikePrice"))
            expiry = opt.get("expiryDate")
            underlying_security = opt.get("underlyingSecurity") or {}
            underlying_stock = underlying_security.get("stock") or {}
            underlying = underlying_stock.get("symbol")
            if not symbol:
                symbol = underlying or ""

        avg_price = node.get("marketAveragePrice") or node.get("averagePrice") or {}
        total_value = node.get("totalValue") or {}
        book = node.get("marketBookValue") or node.get("bookValue") or {}
        unreal = node.get("marketUnrealizedReturns") or node.get("unrealizedReturns") or {}

        accounts: List[str] = [
            str(a.get("id")) for a in (node.get("accounts") or [])
            if isinstance(a, dict) and a.get("id")
        ]

        return cls(
            security_id=security.get("id") or "",
            symbol=symbol,
            quantity=_decimal(node.get("quantity")),
            average_price=_decimal(avg_price.get("amount") if isinstance(avg_price, dict) else avg_price),
            market_value=_decimal(total_value.get("amount") if isinstance(total_value, dict) else total_value),
            book_value=_decimal(book.get("amount") if isinstance(book, dict) else book),
            unrealized_pnl=_decimal(unreal.get("amount") if isinstance(unreal, dict) else unreal),
            is_option=is_option,
            option_type=opt_type,
            strike=strike,
            expiry=expiry,
            underlying_symbol=underlying,
            accounts=accounts,
            raw=dict(node),
        )


# ================================================================== Balance

@dataclass(frozen=True)
class Balance:
    net_liquidation: Decimal
    net_deposits: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    currency: str = "USD"
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_financials(cls, current: dict) -> "Balance":
        current = current or {}
        nlv = (current.get("netLiquidationValueV2") or {})
        deps = (current.get("netDeposits") or {})
        simple = (current.get("simpleReturns") or {})
        ret_amount = (simple.get("amount") or {})
        return cls(
            net_liquidation=_decimal(nlv.get("amount")),
            net_deposits=_decimal(deps.get("amount")),
            total_return=_decimal(ret_amount.get("amount")),
            total_return_pct=_decimal(simple.get("rate")) * Decimal("100"),
            currency=nlv.get("currency") or "USD",
            raw=dict(current),
        )
