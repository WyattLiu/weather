"""Order placement, cancellation, and reliable status polling.

The key addition here is :func:`fetch_extended_order` plus
:func:`wait_for_order`: instead of tailing the activity feed for fills
(which is laggy and prone to false positives on transient 401s), we hit
the ``soOrdersExtendedOrder`` GraphQL query directly by external ID and
get the canonical status, fill quantity, and average fill price.

All the legacy placement/cancel helpers are preserved with the same
signatures so downstream scripts keep working through the ``ws_trading``
compat shim.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

import requests

from .errors import OrderNotFound, OrderTimeout
from .gql import graphql_query
from .models import MultilegOrder, Order
from .queries import (
    MUTATION_ORDER_CANCEL,
    MUTATION_ORDER_CREATE,
    MUTATION_ORDER_EXECUTION_CREATE,
    MUTATION_ORDER_MODIFY,
    MUTATION_PREFLIGHT_CHECK,
    QUERY_EXTENDED_ORDER,
    QUERY_MULTILEG_ORDER,
)

# Keep the legacy constant exported so existing scripts keep working.
# The margin account discovery in ws_sdk.accounts overrides this at runtime.
DEFAULT_ACCOUNT_ID = "non-registered-BpgPfFs0QA"

BRANCH_ID = "TR"   # Wealthsimple Trade branch identifier


# ------------------------------------------------------------------ helpers
def generate_order_id() -> str:
    """Return a unique external order ID (``order-<uuid4>``)."""
    return f"order-{uuid.uuid4()}"


# ==================================================================== place
def place_order(
    session: requests.Session,
    order_type: str,
    security_id: str,
    quantity: int,
    limit_price: float,
    time_in_force: str = "DAY",
    open_close: Optional[str] = None,
    account_id: Optional[str] = None,
) -> dict:
    """Place a single-leg limit order.

    Returns a dict ``{"order_id": str, "result": <graphql_response>}``.
    The legacy signature is preserved for backward compat: an ``account_id``
    kwarg was added so the new :class:`WSClient` can route orders to the
    discovered margin account, but callers that don't pass it still get
    :data:`DEFAULT_ACCOUNT_ID`.
    """
    order_id = generate_order_id()
    acct = account_id or DEFAULT_ACCOUNT_ID

    input_data: dict = {
        "canonicalAccountId": acct,
        "externalId": order_id,
        "executionType": "LIMIT",
        "orderType": order_type,   # BUY_QUANTITY or SELL_QUANTITY
        "quantity": quantity,
        "securityId": security_id,
        "timeInForce": time_in_force,
        "limitPrice": limit_price,
    }

    if security_id.startswith("sec-s-"):
        input_data["tradingSession"] = "ALL"

    if open_close and security_id.startswith("sec-o-"):
        input_data["openClose"] = open_close

    result = graphql_query(
        session,
        "SoOrdersOrderCreate",
        MUTATION_ORDER_CREATE,
        {"input": input_data},
    )
    return {"order_id": order_id, "result": result}


def cancel_order(session: requests.Session, order_id: str) -> dict:
    """Cancel an order by its external ID."""
    return graphql_query(
        session,
        "SoOrdersOrderCancel",
        MUTATION_ORDER_CANCEL,
        {"cancelOrderRequest": {"externalId": order_id}},
    )


def modify_order(
    session: requests.Session, order_id: str, new_limit_price: float
) -> dict:
    """Modify an order's limit price."""
    return graphql_query(
        session,
        "SoOrdersOrderModify",
        MUTATION_ORDER_MODIFY,
        {"input": {"externalId": order_id, "newLimitPrice": new_limit_price}},
    )


def preflight_multileg(
    session: requests.Session,
    account_id: str,
    activity_id: str,
    orders: list,
    net_cash_delta: float,
) -> dict:
    """Run the pre-flight check for a multi-leg order.

    ``orders`` is a list of dicts with keys:
    ``orderId``, ``side`` (BUY/SELL), ``securityId``, ``quantity``,
    ``openClose`` (optional). ``net_cash_delta`` is negative for debit,
    positive for credit.
    """
    preflight_orders = []
    for o in orders:
        leg: dict = {
            "orderId": o["orderId"],
            "side": o["side"],
            "assetLeg": {
                "quantity": o["quantity"],
                "securityId": o["securityId"],
            },
        }
        if o.get("openClose"):
            leg["optionInfo"] = {"openClose": o["openClose"]}
        preflight_orders.append(leg)

    return graphql_query(
        session,
        "ActivityPreFlightCheck",
        MUTATION_PREFLIGHT_CHECK,
        {
            "input": {
                "accountId": account_id,
                "activityId": activity_id,
                "activity": {
                    "multiLegOrder": {
                        "orderBatchId": activity_id,
                        "orders": preflight_orders,
                        "netCash": {
                            "delta": net_cash_delta,
                            "securityId": "sec-c-usd",
                        },
                    }
                },
            }
        },
    )


def place_multileg_order(
    session: requests.Session,
    legs: list,
    limit_price: float,
    quantity_multiplier: int = 1,
    time_in_force: str = "DAY",
    account_id: Optional[str] = None,
) -> dict:
    """Place a multi-leg option order (spread, straddle, roll).

    ``legs`` is a list of dicts with keys ``securityId``, ``orderType``
    (``BUY_QUANTITY``/``SELL_QUANTITY``), and ``openClose``
    (``OPEN``/``CLOSE``). ``limit_price`` is the net per-contract
    price: positive for debit, negative for credit.
    """
    order_uuid = str(uuid.uuid4())
    acct = account_id or DEFAULT_ACCOUNT_ID

    execution_orders = []
    for leg in legs:
        execution_orders.append(
            {
                "openClose": leg.get("openClose", "OPEN"),
                "orderType": leg["orderType"],
                "quantity": 1,
                "securityId": leg["securityId"],
            }
        )

    result = graphql_query(
        session,
        "SoOrdersOrderExecutionCreate",
        MUTATION_ORDER_EXECUTION_CREATE,
        {
            "input": {
                "canonicalAccountId": acct,
                "executionType": "LIMIT",
                "externalId": f"order-{order_uuid}",
                "limitPrice": limit_price,
                "orders": execution_orders,
                "quantityMultiplier": quantity_multiplier,
                "timeInForce": time_in_force,
            }
        },
    )

    return {"order_id": f"order-{order_uuid}", "result": result}


# ==================================================================== status

def fetch_multileg_order(
    session: requests.Session, order_batch_id: str
) -> dict:
    """Fetch the raw multi-leg order payload by batch ID."""
    return graphql_query(
        session,
        "FetchSoOrdersMultilegOrder",
        QUERY_MULTILEG_ORDER,
        {"branchId": BRANCH_ID, "orderBatchId": order_batch_id},
    )


def fetch_extended_order(
    session: requests.Session, external_id: str
) -> Optional[Order]:
    """Fetch a single-leg order by external ID and normalise to :class:`Order`.

    Returns ``None`` if the order is not (yet) visible to WS. This is
    the authoritative source of order status — use it instead of
    polling the activity feed.
    """
    data = graphql_query(
        session,
        "FetchSoOrdersExtendedOrder",
        QUERY_EXTENDED_ORDER,
        {"branchId": BRANCH_ID, "externalId": external_id},
    )
    raw = (data or {}).get("soOrdersExtendedOrder")
    if not raw:
        return None
    return Order.from_extended(raw, external_id=external_id)


def wait_for_order(
    session: requests.Session,
    external_id: str,
    timeout: float = 180.0,
    poll_interval: float = 2.0,
    sleep: Optional[callable] = None,  # type: ignore[type-arg]
    now: Optional[callable] = None,    # type: ignore[type-arg]
) -> Order:
    """Block until an order reaches a terminal state, or time out.

    Calls :func:`fetch_extended_order` every ``poll_interval`` seconds.
    Returns the final :class:`Order` the moment it becomes terminal.

    Raises :class:`OrderTimeout` with the last known snapshot attached if
    the deadline passes while the order is still working, or
    :class:`OrderNotFound` if the API never returns any state for the
    external ID.

    The ``sleep`` and ``now`` hooks exist purely for deterministic tests
    (they default to :func:`time.sleep` / :func:`time.time`).
    """
    _sleep = sleep or time.sleep
    _now = now or time.time

    deadline = _now() + timeout
    last: Optional[Order] = None

    while True:
        order = fetch_extended_order(session, external_id)
        if order is not None:
            last = order
            if order.is_terminal:
                return order

        if _now() >= deadline:
            break
        _sleep(poll_interval)

    if last is None:
        raise OrderNotFound(external_id)
    raise OrderTimeout(external_id, last_state=last)


def wait_for_multileg_order(
    session: requests.Session,
    order_batch_id: str,
    timeout: float = 180.0,
    poll_interval: float = 2.0,
    sleep: Optional[callable] = None,  # type: ignore[type-arg]
    now: Optional[callable] = None,    # type: ignore[type-arg]
) -> MultilegOrder:
    """Block until a multi-leg order reaches a terminal state, or time out.

    Mirrors :func:`wait_for_order` but uses the multi-leg query.
    """
    _sleep = sleep or time.sleep
    _now = now or time.time

    deadline = _now() + timeout
    last: Optional[MultilegOrder] = None

    while True:
        raw = fetch_multileg_order(session, order_batch_id)
        payload = (raw or {}).get("soOrdersMultilegOrder")
        if payload:
            mleg = MultilegOrder.from_response(payload)
            last = mleg
            if mleg.is_terminal:
                return mleg

        if _now() >= deadline:
            break
        _sleep(poll_interval)

    if last is None:
        raise OrderNotFound(order_batch_id)
    # Reuse OrderTimeout — caller can still introspect last_state (MultilegOrder has same .status)
    raise OrderTimeout(order_batch_id, last_state=None)
