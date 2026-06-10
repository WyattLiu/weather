"""High-level :class:`WSClient` wrapper for the Wealthsimple Trade SDK.

A thin facade around the functional modules in this package. A typical
usage is::

    from ws_sdk import WSClient

    ws = WSClient()
    order = ws.sell_to_open(sec_id, qty=1, price=2.50)
    order = ws.wait_for_order(order.external_id, timeout=180)
    if order.is_filled:
        print(f"Filled at ${order.average_filled_price}")

The client auto-discovers the margin account on first use and routes
every order through it. It also filters list queries so crypto and
recurring orders never leak into results.
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

import requests

from .accounts import DEFAULT_ACCOUNT_ID, fetch_all_accounts, get_margin_account_id
from .auth import (
    extract_identity_from_cookies,
    get_session,
    load_cookies,
)
from .errors import OrderRejected
from .gql import graphql_query
from .models import (
    Account,
    Balance,
    MultilegOrder,
    OpenClose,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from .orders import (
    cancel_order as _cancel_order,
    fetch_extended_order,
    fetch_multileg_order,
    modify_order as _modify_order,
    place_multileg_order as _place_multileg_order,
    place_order as _place_order,
    wait_for_multileg_order,
    wait_for_order,
)
from .positions import fetch_positions
from .queries import (
    QUERY_FETCH_ACTIVITIES,
    QUERY_FETCH_FINANCIALS,
)

# Activity types that represent equity/options orders (not crypto/managed/recurring).
_TRADEABLE_ACTIVITY_TYPES = (
    "DIY_BUY",
    "DIY_SELL",
    "OPTIONS_BUY",
    "OPTIONS_SELL",
    "OPTIONS_MULTILEG",
)

# Activity subtypes that should never appear in "open orders" even if the
# underlying type is tradeable. Recurring orders show up as pending
# indefinitely and confuse order management.
_EXCLUDED_SUBTYPES = (
    "RECURRING_ORDER_UPCOMING",
    "RECURRING_ORDER",
    "AUTO_INVEST",
)

# Terminal (non-open) order statuses, excluded client-side in
# list_open_orders. Server-side unifiedStatuses filtering broke in
# June 2026 when WS removed SUBMITTED/WORKING from the enum.
_TERMINAL_STATUSES = frozenset({
    "COMPLETED", "FILLED", "CANCELLED", "CANCELED",
    "EXPIRED", "REJECTED", "FAILED",
})


class WSClient:
    """High-level client for the Wealthsimple Trade API.

    All order-placement methods return normalised :class:`Order` objects.
    For reliable fill detection, call :meth:`wait_for_order` on the
    returned ``external_id``.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        account_id: Optional[str] = None,
        identity_id: Optional[str] = None,
    ):
        self.session = session or get_session()

        if identity_id is None:
            cookies = load_cookies()
            identity_id = extract_identity_from_cookies(cookies)
        if not identity_id:
            raise RuntimeError(
                "WSClient: could not discover identity_id from cookies. "
                "Re-export cookies from the browser."
            )
        self.identity_id: str = identity_id

        if account_id is None:
            try:
                account_id = get_margin_account_id(self.session, self.identity_id)
            except Exception as exc:
                print(f"[WSClient] margin discovery failed: {exc}")
                account_id = DEFAULT_ACCOUNT_ID
        self.account_id: str = account_id or DEFAULT_ACCOUNT_ID

    # ================================================================== orders

    def _place(
        self,
        side: OrderSide,
        security_id: str,
        qty: int,
        price: float,
        open_close: OpenClose,
        time_in_force: str = "DAY",
    ) -> Order:
        raw = _place_order(
            self.session,
            order_type=side.value,
            security_id=security_id,
            quantity=qty,
            limit_price=price,
            time_in_force=time_in_force,
            open_close=open_close.value if security_id.startswith("sec-o-") else None,
            account_id=self.account_id,
        )
        result = raw.get("result") or {}
        soc = (result.get("soOrdersCreateOrder") or {})
        errors = soc.get("errors") or []
        if errors:
            err = errors[0]
            raise OrderRejected(
                f"Order rejected: {err.get('message', err)}",
                external_id=raw.get("order_id"),
                code=err.get("code"),
            )

        # Synthesize an initial Order snapshot. Full status comes from wait_for_order.
        return Order(
            external_id=raw.get("order_id") or "",
            status=OrderStatus.PENDING,
            side=side,
            open_close=open_close,
            security_id=security_id,
            submitted_quantity=Decimal(str(qty)),
            filled_quantity=Decimal("0"),
            average_filled_price=None,
            limit_price=Decimal(str(price)),
            time_in_force=time_in_force,
            submitted_at=None,
            first_filled_at=None,
            last_filled_at=None,
            expired_at=None,
            rejection_cause=None,
            rejection_code=None,
            canonical_account_id=self.account_id,
            raw=raw,
        )

    def buy_to_open(self, security_id: str, qty: int, price: float) -> Order:
        return self._place(OrderSide.BUY, security_id, qty, price, OpenClose.OPEN)

    def sell_to_open(self, security_id: str, qty: int, price: float) -> Order:
        return self._place(OrderSide.SELL, security_id, qty, price, OpenClose.OPEN)

    def buy_to_close(self, security_id: str, qty: int, price: float) -> Order:
        return self._place(OrderSide.BUY, security_id, qty, price, OpenClose.CLOSE)

    def sell_to_close(self, security_id: str, qty: int, price: float) -> Order:
        return self._place(OrderSide.SELL, security_id, qty, price, OpenClose.CLOSE)

    def place_multileg(
        self,
        legs: list,
        net_price: float,
        qty: int = 1,
        time_in_force: str = "DAY",
    ) -> MultilegOrder:
        """Place a multi-leg order.

        ``legs`` is a list of ``{"securityId", "orderType", "openClose"}``
        dicts. ``net_price`` is per-contract (positive for debit, negative
        for credit). Returns an initial :class:`MultilegOrder` snapshot;
        use :meth:`wait_for_multileg` to get the final state.
        """
        raw = _place_multileg_order(
            self.session,
            legs=legs,
            limit_price=net_price,
            quantity_multiplier=qty,
            time_in_force=time_in_force,
            account_id=self.account_id,
        )
        # Return a minimal snapshot — caller polls with wait_for_multileg.
        return MultilegOrder(
            batch_id=raw.get("order_id") or "",
            external_id=raw.get("order_id") or "",
            status=OrderStatus.PENDING,
            strategy=None,
            limit_price=Decimal(str(net_price)),
            time_in_force=time_in_force,
            submitted_at=None,
            updated_at=None,
            total_fee=None,
            legs=[],
            raw=raw,
        )

    def cancel(self, external_id: str) -> bool:
        """Cancel an order by external ID. Returns True on success."""
        result = _cancel_order(self.session, external_id) or {}
        payload = result.get("orderServiceCancelOrder") or {}
        errors = payload.get("errors") or []
        return not errors

    def modify(self, external_id: str, new_price: float) -> dict:
        return _modify_order(self.session, external_id, new_price)

    def get_order(self, external_id: str) -> Optional[Order]:
        return fetch_extended_order(self.session, external_id)

    def wait_for_order(
        self, external_id: str, timeout: float = 180, poll_interval: float = 2
    ) -> Order:
        return wait_for_order(self.session, external_id, timeout=timeout, poll_interval=poll_interval)

    def get_multileg(self, batch_id: str) -> Optional[MultilegOrder]:
        raw = fetch_multileg_order(self.session, batch_id)
        payload = (raw or {}).get("soOrdersMultilegOrder")
        return MultilegOrder.from_response(payload) if payload else None

    def wait_for_multileg(
        self, batch_id: str, timeout: float = 180, poll_interval: float = 2
    ) -> MultilegOrder:
        return wait_for_multileg_order(
            self.session, batch_id, timeout=timeout, poll_interval=poll_interval
        )

    # ================================================================== reads

    def list_positions(self) -> List[Position]:
        """Return positions held in the margin account only.

        Crypto positions (which live on a separate Wealthsimple account)
        never appear because the underlying query is scoped to
        ``[self.account_id]``.
        """
        return fetch_positions(
            self.session,
            self.identity_id,
            account_ids=[self.account_id],
        )

    def list_open_orders(self, verify: bool = True) -> List[Order]:
        """Return all pending/working orders on the margin account.

        Uses the activity feed as a *discovery* mechanism (filtered
        server-side by account and activity type to exclude crypto and
        recurring noise), then optionally verifies each discovered order
        against :func:`fetch_extended_order` to get the canonical status
        and fill fields.
        """
        # NOTE: no server-side unifiedStatuses filter. WS changed the enum
        # (June 2026): SUBMITTED/WORKING now return UNPROCESSABLE_ENTITY and
        # the whole query silently failed → list_open_orders returned [] →
        # the escalation sweep was a no-op. Filter by status client-side
        # instead (terminal-status exclusion below) — resilient to enum drift.
        data = graphql_query(
            self.session,
            "FetchActivityFeedItems",
            QUERY_FETCH_ACTIVITIES,
            {
                "first": 50,
                "orderBy": "OCCURRED_AT_DESC",
                "condition": {
                    "accountIds": [self.account_id],
                    "types": list(_TRADEABLE_ACTIVITY_TYPES),
                },
            },
        )

        edges = ((data or {}).get("activityFeedItems") or {}).get("edges") or []
        external_ids: list[str] = []
        filtered_edges: list[dict] = []
        for edge in edges:
            node = edge.get("node") or {}

            # Belt-and-suspenders: server-side filters should have done this,
            # but some WS responses ignore condition fields and we can't trust
            # the server. Re-apply the filter client-side.
            if node.get("accountId") and node["accountId"] != self.account_id:
                continue
            node_type = (node.get("type") or "").upper()
            if node_type and node_type not in _TRADEABLE_ACTIVITY_TYPES:
                continue
            sub_type = (node.get("subType") or "").upper()
            if sub_type in _EXCLUDED_SUBTYPES:
                continue
            # Client-side open-state filter (replaces the broken server-side
            # unifiedStatuses condition). Terminal statuses are not open.
            status = (node.get("unifiedStatus") or node.get("status") or "").upper()
            if status in _TERMINAL_STATUSES:
                continue

            ext_id = node.get("externalCanonicalId") or node.get("canonicalId")
            if ext_id:
                external_ids.append(ext_id)
                filtered_edges.append(edge)

        if not verify:
            # Return the activity-feed view as Order-ish stubs. Filter was
            # already applied above when building filtered_edges.
            return [self._activity_to_stub(edge.get("node") or {}) for edge in filtered_edges]

        orders: list[Order] = []
        for ext_id in external_ids:
            order = fetch_extended_order(self.session, ext_id)
            if order is None:
                continue
            if order.is_terminal:
                continue  # Activity feed is laggy — drop already-closed items
            orders.append(order)
        return orders

    def _activity_to_stub(self, node: dict) -> Order:
        """Build a best-effort Order from an activity-feed node (no extended fetch)."""
        raw_type = (node.get("type") or "").upper()
        side: Optional[OrderSide] = None
        if "BUY" in raw_type:
            side = OrderSide.BUY
        elif "SELL" in raw_type:
            side = OrderSide.SELL
        qty_str = node.get("assetQuantity") or "0"
        return Order(
            external_id=node.get("externalCanonicalId") or node.get("canonicalId") or "",
            status=OrderStatus.from_raw(node.get("unifiedStatus") or node.get("status")),
            side=side,
            open_close=None,
            security_id=node.get("securityId") or "",
            submitted_quantity=Decimal(str(qty_str)) if qty_str else Decimal("0"),
            filled_quantity=Decimal("0"),
            average_filled_price=None,
            limit_price=None,
            time_in_force=None,
            submitted_at=None,
            first_filled_at=None,
            last_filled_at=None,
            expired_at=None,
            rejection_cause=None,
            rejection_code=None,
            canonical_account_id=node.get("accountId"),
            raw=dict(node),
        )

    def list_accounts(self) -> List[Account]:
        """Return all accounts on the identity (margin + crypto + cash + ...)."""
        return fetch_all_accounts(self.session, self.identity_id)

    def get_balance(self, currency: str = "USD") -> Balance:
        """Return the current net-liquidation snapshot for the margin account."""
        data = graphql_query(
            self.session,
            "FetchIdentityCurrentFinancials",
            QUERY_FETCH_FINANCIALS,
            {
                "identityId": self.identity_id,
                "currency": currency,
                "startDate": None,
                "accountIds": [self.account_id],
            },
        )
        current = (
            (data or {}).get("identity", {}).get("financials", {}).get("current", {})
        )
        return Balance.from_financials(current)
