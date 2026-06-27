"""Tests for ws_sdk.orders — fetch_extended_order, wait_for_order, placement."""
from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import load_fixture
from ws_sdk.errors import OrderNotFound, OrderTimeout
from ws_sdk.models import OrderStatus
from ws_sdk.orders import (
    fetch_extended_order,
    generate_order_id,
    place_order,
    wait_for_order,
)


# ---------- generate_order_id ----------------------------------------

def test_generate_order_id_unique_and_prefixed():
    a = generate_order_id()
    b = generate_order_id()
    assert a.startswith("order-")
    assert b.startswith("order-")
    assert a != b


# ---------- fetch_extended_order -------------------------------------

def test_fetch_extended_order_filled(fake_session, patch_graphql):
    patch_graphql({
        "FetchSoOrdersExtendedOrder": load_fixture("extended_order_filled"),
    })
    order = fetch_extended_order(fake_session, "order-be620327-74fb-43af-a09c-2a15b58a6ed7")
    assert order is not None
    assert order.is_filled
    assert order.average_filled_price == Decimal("1.7100")


def test_fetch_extended_order_missing_returns_none(fake_session, patch_graphql):
    patch_graphql({"FetchSoOrdersExtendedOrder": {"soOrdersExtendedOrder": None}})
    order = fetch_extended_order(fake_session, "order-nonexistent")
    assert order is None


def test_fetch_extended_order_empty_response(fake_session, patch_graphql):
    patch_graphql({"FetchSoOrdersExtendedOrder": {}})
    order = fetch_extended_order(fake_session, "order-nonexistent")
    assert order is None


# ---------- wait_for_order (the killer feature) ----------------------

def test_wait_for_order_returns_immediately_on_terminal(fake_session, patch_graphql, fake_clock):
    patch_graphql({
        "FetchSoOrdersExtendedOrder": load_fixture("extended_order_filled"),
    })
    order = wait_for_order(
        fake_session, "order-1",
        timeout=60, poll_interval=2,
        sleep=fake_clock.sleep, now=fake_clock.now,
    )
    assert order.is_filled
    # No sleep needed because first poll returned terminal
    assert fake_clock.sleeps == []


def test_wait_for_order_polls_multiple_times_then_succeeds(fake_session, patch_graphql, fake_clock):
    pending = load_fixture("extended_order_pending")
    filled = load_fixture("extended_order_filled")

    call_count = {"n": 0}

    def responder(op_name, _vars):
        if op_name != "FetchSoOrdersExtendedOrder":
            return {}
        call_count["n"] += 1
        if call_count["n"] < 3:
            return pending
        return filled

    patch_graphql(responder)
    order = wait_for_order(
        fake_session, "order-1",
        timeout=60, poll_interval=2,
        sleep=fake_clock.sleep, now=fake_clock.now,
    )
    assert order.is_filled
    assert call_count["n"] == 3
    assert len(fake_clock.sleeps) == 2  # 2 sleeps between 3 polls


def test_wait_for_order_times_out_with_last_state(fake_session, patch_graphql, fake_clock):
    pending = load_fixture("extended_order_pending")
    patch_graphql({"FetchSoOrdersExtendedOrder": pending})

    with pytest.raises(OrderTimeout) as exc:
        wait_for_order(
            fake_session, "order-pending",
            timeout=10, poll_interval=2,
            sleep=fake_clock.sleep, now=fake_clock.now,
        )
    err = exc.value
    assert err.external_id == "order-pending"
    assert err.last_state is not None
    assert err.last_state.status is OrderStatus.PENDING


def test_wait_for_order_not_found_if_never_visible(fake_session, patch_graphql, fake_clock):
    patch_graphql({"FetchSoOrdersExtendedOrder": {"soOrdersExtendedOrder": None}})

    with pytest.raises(OrderNotFound) as exc:
        wait_for_order(
            fake_session, "order-ghost",
            timeout=5, poll_interval=1,
            sleep=fake_clock.sleep, now=fake_clock.now,
        )
    assert exc.value.external_id == "order-ghost"


def test_wait_for_order_returns_on_rejection(fake_session, patch_graphql, fake_clock):
    patch_graphql({
        "FetchSoOrdersExtendedOrder": load_fixture("extended_order_rejected"),
    })
    order = wait_for_order(
        fake_session, "order-rejected",
        timeout=60, poll_interval=1,
        sleep=fake_clock.sleep, now=fake_clock.now,
    )
    assert order.status is OrderStatus.REJECTED
    assert order.is_terminal
    assert not order.is_filled
    assert order.rejection_cause == "INSUFFICIENT_BUYING_POWER"


# ---------- place_order ----------------------------------------------

def test_place_order_forwards_account_id(fake_session, patch_graphql):
    captured = {}

    def responder(op_name, variables):
        if op_name == "SoOrdersOrderCreate":
            captured.update(variables)
            return {"soOrdersCreateOrder": {"errors": [], "order": {"orderId": "server-1"}}}
        return {}

    patch_graphql(responder)

    result = place_order(
        fake_session,
        order_type="SELL_QUANTITY",
        security_id="sec-o-abc",
        quantity=1,
        limit_price=2.50,
        open_close="OPEN",
        account_id="non-registered-BpgPfFs0QA",
    )
    assert result["order_id"].startswith("order-")
    # The GraphQL input must have flowed through to the mutation
    inp = captured["input"]
    assert inp["canonicalAccountId"] == "non-registered-BpgPfFs0QA"
    assert inp["orderType"] == "SELL_QUANTITY"
    assert inp["openClose"] == "OPEN"
    assert inp["securityId"] == "sec-o-abc"
    assert inp["quantity"] == 1
    assert inp["limitPrice"] == 2.50


def test_place_order_stock_adds_trading_session(fake_session, patch_graphql):
    captured = {}

    def responder(op_name, variables):
        if op_name == "SoOrdersOrderCreate":
            captured.update(variables)
            return {"soOrdersCreateOrder": {"errors": [], "order": {"orderId": "server-1"}}}
        return {}

    patch_graphql(responder)

    place_order(
        fake_session,
        order_type="BUY_QUANTITY",
        security_id="sec-s-ung",  # stock
        quantity=100,
        limit_price=12.50,
    )
    inp = captured["input"]
    assert inp["tradingSession"] == "ALL"
    assert "openClose" not in inp   # stock orders don't carry openClose
