"""Backward-compatibility shim test.

Every name that a downstream script ever imported from ``ws_trading``
must still resolve. The audit of consumers (ws_ung_strangle, spy_scalp,
ws_swap, ws_ung_grid, ws_ung_accumulator, spy_ladder, iv_term_structure,
intc_rc_analyzer, etc.) produced the name list below — this test locks
it in.
"""
from __future__ import annotations


def test_legacy_ws_trading_surface_intact():
    # Individual imports so a failure tells you exactly which name broke.
    from ws_trading import get_session  # noqa: F401
    from ws_trading import graphql_query  # noqa: F401
    from ws_trading import place_order  # noqa: F401
    from ws_trading import cancel_order  # noqa: F401
    from ws_trading import modify_order  # noqa: F401
    from ws_trading import place_multileg_order  # noqa: F401
    from ws_trading import fetch_multileg_order  # noqa: F401
    from ws_trading import preflight_multileg  # noqa: F401
    from ws_trading import generate_order_id  # noqa: F401

    from ws_trading import load_config  # noqa: F401
    from ws_trading import save_config  # noqa: F401
    from ws_trading import load_cookies  # noqa: F401
    from ws_trading import save_cookies  # noqa: F401
    from ws_trading import extract_oauth_data  # noqa: F401
    from ws_trading import is_token_expired  # noqa: F401
    from ws_trading import refresh_access_token  # noqa: F401
    from ws_trading import update_cookies_with_new_token  # noqa: F401
    from ws_trading import extract_access_token  # noqa: F401
    from ws_trading import extract_identity_from_cookies  # noqa: F401
    from ws_trading import extract_accounts_from_cookies  # noqa: F401

    from ws_trading import QUERY_FETCH_FINANCIALS  # noqa: F401
    from ws_trading import QUERY_FETCH_POSITIONS  # noqa: F401
    from ws_trading import QUERY_FETCH_ACTIVITIES  # noqa: F401
    from ws_trading import QUERY_FETCH_SECURITY  # noqa: F401
    from ws_trading import QUERY_SECURITY_SEARCH  # noqa: F401
    from ws_trading import QUERY_OPTION_EXPIRATION_DATES  # noqa: F401
    from ws_trading import QUERY_OPTION_CHAIN  # noqa: F401
    from ws_trading import QUERY_MULTILEG_ORDER  # noqa: F401
    from ws_trading import MUTATION_ORDER_CREATE  # noqa: F401
    from ws_trading import MUTATION_ORDER_CANCEL  # noqa: F401
    from ws_trading import MUTATION_ORDER_MODIFY  # noqa: F401
    from ws_trading import MUTATION_ORDER_EXECUTION_CREATE  # noqa: F401
    from ws_trading import MUTATION_PREFLIGHT_CHECK  # noqa: F401

    from ws_trading import KNOWN_SECURITIES
    from ws_trading import DEFAULT_ACCOUNT_ID

    # Quick sanity on data shape
    assert "UNG" in KNOWN_SECURITIES
    assert "SPY" in KNOWN_SECURITIES
    assert DEFAULT_ACCOUNT_ID.startswith("non-registered-")


def test_new_sdk_surface_also_exposed_via_ws_trading():
    """New SDK abstractions are reachable through the shim too."""
    from ws_trading import WSClient  # noqa: F401
    from ws_trading import OrderStatus  # noqa: F401
    from ws_trading import OrderSide  # noqa: F401
    from ws_trading import AccountType  # noqa: F401
    from ws_trading import wait_for_order  # noqa: F401
    from ws_trading import fetch_extended_order  # noqa: F401
    from ws_trading import get_margin_account_id  # noqa: F401
    from ws_trading import OrderTimeout, OrderNotFound, OrderRejected  # noqa: F401
    from ws_trading import QUERY_EXTENDED_ORDER, QUERY_ALL_ACCOUNTS  # noqa: F401

    assert "FetchSoOrdersExtendedOrder" in QUERY_EXTENDED_ORDER
    assert "FetchAllAccounts" in QUERY_ALL_ACCOUNTS


def test_ws_sdk_package_imports():
    """Ensure the canonical `from ws_sdk import ...` path works."""
    from ws_sdk import (
        WSClient,
        Order,
        Position,
        Account,
        Balance,
        OrderStatus,
        OrderSide,
        OpenClose,
        AccountType,
        wait_for_order,
        fetch_extended_order,
        get_margin_account_id,
        KNOWN_SECURITIES,
        DEFAULT_ACCOUNT_ID,
    )
    assert WSClient is not None
    assert Order is not None
    assert OrderStatus is not None
