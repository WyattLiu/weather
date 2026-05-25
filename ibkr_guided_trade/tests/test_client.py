"""Tests for the WSClient — the user-facing high-level interface.

The critical test here is :func:`test_list_open_orders_filters_crypto`:
the whole point of this refactor is that crypto activities and
recurring orders must never appear in the margin account's open-orders
view.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import load_fixture
from ws_sdk.accounts import reset_margin_cache
from ws_sdk.client import WSClient
from ws_sdk.models import OrderStatus


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_margin_cache()
    yield
    reset_margin_cache()


@pytest.fixture
def ws_client(fake_session, patch_graphql):
    """A WSClient pre-loaded with account discovery + a margin account."""
    # Seed the fetch_all_accounts + identity resolution paths so __init__ works.
    patch_graphql({
        "FetchAllAccounts": load_fixture("all_accounts"),
    })
    # Avoid touching real cookies in __init__
    with patch("ws_sdk.client.get_session", return_value=fake_session), \
         patch("ws_sdk.client.load_cookies", return_value={}), \
         patch("ws_sdk.client.extract_identity_from_cookies", return_value="identity-test"):
        ws = WSClient()
    return ws


def test_wsclient_discovers_margin_account(ws_client):
    assert ws_client.account_id == "non-registered-BpgPfFs0QA"
    assert ws_client.identity_id == "identity-test"


def test_list_open_orders_filters_crypto_and_recurring(ws_client, patch_graphql):
    """The main regression test for this whole refactor.

    Given an activity feed with crypto orders, recurring orders, managed
    robo orders, and one legit margin limit order, list_open_orders
    must return exactly the margin limit order.
    """
    # Verified extended-order response for the legit margin orders
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            return load_fixture("activity_feed_mixed")
        if op_name == "FetchSoOrdersExtendedOrder":
            # Return a WORKING state for each verified order so nothing drops
            ext_id = variables.get("externalId") or ""
            return {
                "soOrdersExtendedOrder": {
                    "externalId": ext_id,
                    "status": "pending",
                    "orderType": "BUY_QUANTITY",
                    "openClose": "OPEN",
                    "securityId": "sec-o-test",
                    "submittedQuantity": "1",
                    "filledQuantity": "0",
                    "limitPrice": "12.50",
                    "canonicalAccountId": "non-registered-BpgPfFs0QA",
                }
            }
        return {}

    patch_graphql(responder)
    open_orders = ws_client.list_open_orders()

    # Expectations: the fixture has 5 entries:
    #   1. crypto-recurring       (CRYPTO_BUY + RECURRING_ORDER_UPCOMING — filtered by type)
    #   2. crypto-limit           (CRYPTO_BUY — filtered by type)
    #   3. diy limit (UNG shares) (DIY_BUY + LIMIT_ORDER — SHOULD APPEAR)
    #   4. options sell (UNG)     (OPTIONS_SELL + LIMIT_ORDER — SHOULD APPEAR)
    #   5. managed recurring      (MANAGED_BUY + RECURRING_ORDER — filtered by subtype)
    # Net: 2 orders survive.
    ids = [o.external_id for o in open_orders]
    assert "order-crypto-recurring-1" not in ids
    assert "order-crypto-2" not in ids
    assert "order-managed-1" not in ids
    assert "order-diy-1" in ids
    assert "order-opt-1" in ids
    assert len(open_orders) == 2


def test_list_open_orders_skips_verified_terminal(ws_client, patch_graphql):
    """Activity feed may show orders that have already filled.

    The verify-step must drop them before returning.
    """
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            # Single pending DIY order on the margin account
            return {
                "activityFeedItems": {
                    "edges": [{
                        "node": {
                            "accountId": "non-registered-BpgPfFs0QA",
                            "type": "DIY_BUY",
                            "subType": "LIMIT_ORDER",
                            "unifiedStatus": "PENDING",
                            "externalCanonicalId": "order-stale",
                            "canonicalId": "order-stale",
                            "assetSymbol": "UNG",
                        },
                    }],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        if op_name == "FetchSoOrdersExtendedOrder":
            return load_fixture("extended_order_filled")
        return {}

    patch_graphql(responder)
    open_orders = ws_client.list_open_orders()
    # Activity feed claimed it was pending but extended-order says POSTED.
    # The truth wins — nothing returned.
    assert open_orders == []


def test_list_positions_scoped_to_margin(ws_client, patch_graphql):
    """list_positions must pass accountIds=[margin_id] to the GraphQL call."""
    captured = {}

    def responder(op_name, variables):
        if op_name == "FetchIdentityPositions":
            captured.update(variables)
            return load_fixture("positions_margin")
        return {}

    patch_graphql(responder)
    positions = ws_client.list_positions()

    assert captured.get("accountIds") == ["non-registered-BpgPfFs0QA"]
    assert len(positions) == 3
    ung_shares = [p for p in positions if p.symbol == "UNG" and not p.is_option]
    assert len(ung_shares) == 1
    ung_calls = [p for p in positions if p.is_option and p.option_type == "CALL"]
    assert len(ung_calls) == 1


def test_get_balance_uses_margin_account(ws_client, patch_graphql):
    captured = {}

    def responder(op_name, variables):
        if op_name == "FetchIdentityCurrentFinancials":
            captured.update(variables)
            return {
                "identity": {
                    "id": "identity-test",
                    "financials": {
                        "current": {
                            "id": "c-1",
                            "netLiquidationValueV2": {"amount": "102854.63", "currency": "USD"},
                            "netDeposits": {"amount": "81571.60", "currency": "USD"},
                            "simpleReturns": {
                                "amount": {"amount": "21283.03", "currency": "USD"},
                                "rate": "0.2609",
                            },
                        }
                    }
                }
            }
        return {}

    patch_graphql(responder)
    bal = ws_client.get_balance()
    from decimal import Decimal
    assert bal.net_liquidation == Decimal("102854.63")
    assert captured["accountIds"] == ["non-registered-BpgPfFs0QA"]
