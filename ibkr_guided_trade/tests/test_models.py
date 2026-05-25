"""Tests for ws_sdk.models — typed primitives and classmethod builders."""
from __future__ import annotations

from decimal import Decimal

import pytest

from tests.conftest import load_fixture
from ws_sdk.models import (
    Account,
    AccountType,
    Balance,
    MultilegOrder,
    OpenClose,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)


# ---------- OrderStatus -----------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("posted",      OrderStatus.POSTED),
    ("filled",      OrderStatus.POSTED),
    ("completed",   OrderStatus.POSTED),
    ("POSTED",      OrderStatus.POSTED),
    ("pending",     OrderStatus.PENDING),
    ("submitted",   OrderStatus.SUBMITTED),
    ("accepted",    OrderStatus.SUBMITTED),
    ("working",     OrderStatus.WORKING),
    ("cancelled",   OrderStatus.CANCELLED),
    ("canceled",    OrderStatus.CANCELLED),
    ("rejected",    OrderStatus.REJECTED),
    ("failed",      OrderStatus.REJECTED),
    ("expired",     OrderStatus.EXPIRED),
    ("partially_filled", OrderStatus.WORKING),
    ("",            OrderStatus.PENDING),
    (None,          OrderStatus.PENDING),
    ("wat",         OrderStatus.PENDING),  # unknown falls back to PENDING
])
def test_order_status_from_raw(raw, expected):
    assert OrderStatus.from_raw(raw) is expected


def test_order_status_is_terminal():
    assert OrderStatus.POSTED.is_terminal
    assert OrderStatus.CANCELLED.is_terminal
    assert OrderStatus.REJECTED.is_terminal
    assert OrderStatus.EXPIRED.is_terminal
    assert not OrderStatus.PENDING.is_terminal
    assert not OrderStatus.SUBMITTED.is_terminal
    assert not OrderStatus.WORKING.is_terminal


def test_order_status_is_filled():
    assert OrderStatus.POSTED.is_filled
    assert not OrderStatus.CANCELLED.is_filled
    assert not OrderStatus.REJECTED.is_filled
    assert not OrderStatus.WORKING.is_filled


# ---------- OrderSide / OpenClose -------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("BUY_QUANTITY",  OrderSide.BUY),
    ("BUY",           OrderSide.BUY),
    ("buy",           OrderSide.BUY),
    ("SELL_QUANTITY", OrderSide.SELL),
    ("SELL",          OrderSide.SELL),
    ("",              None),
    (None,            None),
    ("nope",          None),
])
def test_order_side_from_raw(raw, expected):
    assert OrderSide.from_raw(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("OPEN",  OpenClose.OPEN),
    ("open",  OpenClose.OPEN),
    ("CLOSE", OpenClose.CLOSE),
    ("close", OpenClose.CLOSE),
    ("",      None),
    (None,    None),
    ("wat",   None),
])
def test_open_close_from_raw(raw, expected):
    assert OpenClose.from_raw(raw) == expected


# ---------- AccountType -----------------------------------------------

@pytest.mark.parametrize("acct_id, expected", [
    ("non-registered-BpgPfFs0QA",        AccountType.MARGIN),
    ("non-registered-crypto-UCLFqN_t7Q", AccountType.CRYPTO),
    ("ca-cash-msb-HuMPL9AHng",           AccountType.CASH),
    ("ca-credit-card-BiRdg7XZTQ",        AccountType.CREDIT_CARD),
    ("tfsa-ABC123",                      AccountType.TFSA),
    ("rrsp-XYZ456",                      AccountType.RRSP),
    ("",                                 AccountType.OTHER),
    (None,                               AccountType.OTHER),
])
def test_account_type_from_id(acct_id, expected):
    assert AccountType.from_account_id(acct_id) is expected


def test_account_type_crypto_substring_wins_over_non_registered():
    """A non-registered-crypto-X id must classify as CRYPTO, not MARGIN."""
    assert AccountType.from_account_id("non-registered-crypto-UCLFqN_t7Q") == AccountType.CRYPTO


# ---------- Order.from_extended ---------------------------------------

def test_order_from_extended_filled():
    payload = load_fixture("extended_order_filled")["soOrdersExtendedOrder"]
    order = Order.from_extended(payload)

    assert order.external_id == "order-be620327-74fb-43af-a09c-2a15b58a6ed7"
    assert order.status is OrderStatus.POSTED
    assert order.is_filled
    assert order.is_terminal
    assert not order.is_partially_filled
    assert order.side is OrderSide.SELL
    assert order.open_close is OpenClose.OPEN
    assert order.submitted_quantity == Decimal("1.0000")
    assert order.filled_quantity == Decimal("1.0000")
    assert order.average_filled_price == Decimal("1.7100")
    assert order.limit_price == Decimal("1.7100")
    assert order.canonical_account_id == "non-registered-BpgPfFs0QA"
    assert order.submitted_at is not None
    assert order.first_filled_at is not None


def test_order_from_extended_pending():
    payload = load_fixture("extended_order_pending")["soOrdersExtendedOrder"]
    order = Order.from_extended(payload)

    assert order.status is OrderStatus.PENDING
    assert not order.is_filled
    assert not order.is_terminal
    assert order.filled_quantity == Decimal("0.0000")
    assert order.average_filled_price is None
    assert order.limit_price == Decimal("2.50")
    assert order.remaining_quantity == Decimal("1.0000")


def test_order_from_extended_partial():
    payload = load_fixture("extended_order_partial")["soOrdersExtendedOrder"]
    order = Order.from_extended(payload)

    assert order.status is OrderStatus.WORKING
    assert not order.is_terminal
    assert not order.is_filled
    assert order.is_partially_filled
    assert order.submitted_quantity == Decimal("3.0000")
    assert order.filled_quantity == Decimal("1.0000")
    assert order.remaining_quantity == Decimal("2.0000")


def test_order_from_extended_rejected():
    payload = load_fixture("extended_order_rejected")["soOrdersExtendedOrder"]
    order = Order.from_extended(payload)

    assert order.status is OrderStatus.REJECTED
    assert order.is_terminal
    assert not order.is_filled
    assert order.rejection_cause == "INSUFFICIENT_BUYING_POWER"
    assert order.rejection_code == "BP_REJECT"


def test_order_from_extended_empty_is_safe():
    order = Order.from_extended({})
    assert order.status is OrderStatus.PENDING
    assert order.external_id == ""
    assert order.submitted_quantity == Decimal("0")


# ---------- MultilegOrder ---------------------------------------------

def test_multileg_from_response_terminal_when_all_legs_filled():
    payload = {
        "orderBatchId": "order-batch-1",
        "externalId": "order-batch-1",
        "status": "working",     # batch still "working" but all legs posted
        "optionStrategy": "STRADDLE",
        "limitPrice": "10.00",
        "legs": [
            {
                "externalId": "order-batch-1-leg-1",
                "status": "posted",
                "side": "BUY",
                "openClose": "OPEN",
                "securityId": "sec-o-a",
                "submittedQuantity": "1",
                "filledQuantity": "1",
                "averageFillPrice": {"amount": "5.00", "currency": "USD"},
            },
            {
                "externalId": "order-batch-1-leg-2",
                "status": "posted",
                "side": "BUY",
                "openClose": "OPEN",
                "securityId": "sec-o-b",
                "submittedQuantity": "1",
                "filledQuantity": "1",
                "averageFillPrice": {"amount": "5.00", "currency": "USD"},
            },
        ],
    }
    mleg = MultilegOrder.from_response(payload)

    assert len(mleg.legs) == 2
    assert mleg.legs[0].is_filled
    assert mleg.legs[1].is_filled
    # Batch status is "working" but all legs are terminal — is_terminal True
    assert mleg.is_terminal
    assert mleg.is_filled


def test_multileg_pending_not_terminal():
    payload = {
        "orderBatchId": "order-batch-2",
        "externalId": "order-batch-2",
        "status": "pending",
        "legs": [
            {"externalId": "l1", "status": "pending", "side": "BUY", "openClose": "OPEN",
             "securityId": "s-a", "submittedQuantity": "1", "filledQuantity": "0"},
            {"externalId": "l2", "status": "pending", "side": "BUY", "openClose": "OPEN",
             "securityId": "s-b", "submittedQuantity": "1", "filledQuantity": "0"},
        ],
    }
    mleg = MultilegOrder.from_response(payload)
    assert not mleg.is_terminal
    assert not mleg.is_filled


# ---------- Account ---------------------------------------------------

def test_account_from_node_margin():
    node = {
        "id": "non-registered-BpgPfFs0QA",
        "type": "NON_REGISTERED",
        "unifiedAccountType": "NON_REGISTERED",
        "nickname": "Wyatt - Personal",
        "currency": "USD",
        "status": "OPEN",
    }
    acct = Account.from_node(node)
    assert acct.is_margin
    assert acct.type is AccountType.MARGIN
    assert acct.nickname == "Wyatt - Personal"


def test_account_from_node_crypto_not_margin():
    node = {"id": "non-registered-crypto-X", "currency": "CAD"}
    acct = Account.from_node(node)
    assert not acct.is_margin
    assert acct.type is AccountType.CRYPTO


# ---------- Position --------------------------------------------------

def test_position_from_v2_shares():
    node = {
        "id": "pos-1",
        "quantity": "500",
        "security": {
            "id": "sec-s-ung",
            "stock": {"symbol": "UNG"},
        },
        "marketAveragePrice": {"amount": "12.05"},
        "totalValue": {"amount": "6500.00"},
        "marketBookValue": {"amount": "6025.00"},
        "marketUnrealizedReturns": {"amount": "475.00"},
        "accounts": [{"id": "non-registered-BpgPfFs0QA"}],
    }
    pos = Position.from_position_v2(node)
    assert not pos.is_option
    assert pos.symbol == "UNG"
    assert pos.quantity == Decimal("500")
    assert pos.average_price == Decimal("12.05")
    assert pos.accounts == ["non-registered-BpgPfFs0QA"]


def test_position_from_v2_option():
    node = {
        "id": "pos-2",
        "quantity": "-2",
        "security": {
            "id": "sec-o-ung-call",
            "optionDetails": {
                "strikePrice": "13.00",
                "optionType": "CALL",
                "expiryDate": "2026-05-15",
                "underlyingSecurity": {"stock": {"symbol": "UNG"}},
            },
        },
        "marketAveragePrice": {"amount": "0.55"},
        "totalValue": {"amount": "-120.00"},
        "marketBookValue": {"amount": "-110.00"},
        "marketUnrealizedReturns": {"amount": "-10.00"},
        "accounts": [],
    }
    pos = Position.from_position_v2(node)
    assert pos.is_option
    assert pos.option_type == "CALL"
    assert pos.strike == Decimal("13.00")
    assert pos.expiry == "2026-05-15"
    assert pos.underlying_symbol == "UNG"
    assert pos.quantity == Decimal("-2")


# ---------- Balance ---------------------------------------------------

def test_balance_from_financials():
    current = {
        "netLiquidationValueV2": {"amount": "102854.63", "currency": "USD"},
        "netDeposits": {"amount": "81571.60", "currency": "USD"},
        "simpleReturns": {
            "amount": {"amount": "21283.03", "currency": "USD"},
            "rate": "0.2609",
        },
    }
    bal = Balance.from_financials(current)
    assert bal.net_liquidation == Decimal("102854.63")
    assert bal.net_deposits == Decimal("81571.60")
    assert bal.total_return == Decimal("21283.03")
    # Rate is scaled to percent
    assert abs(bal.total_return_pct - Decimal("26.09")) < Decimal("0.01")
