"""Wealthsimple Trade SDK.

High-level Python client for the Wealthsimple Trade GraphQL API. Provides
typed models for orders, positions, accounts, and balances plus a
:class:`WSClient` wrapper that handles auth, account discovery, and
reliable order polling via the ``soOrdersExtendedOrder`` query.

Basic usage::

    from ws_sdk import WSClient, OrderStatus

    ws = WSClient()                             # auto-discovers margin account
    order = ws.sell_to_open(sec_id, 1, 2.50)    # returns Order snapshot
    order = ws.wait_for_order(order.external_id, timeout=180)
    if order.is_filled:
        print(f"Filled at ${order.average_filled_price}")

The package is fully backward-compatible with ``ws_trading`` imports:
every function and query constant that used to live in ``ws_trading.py``
is re-exported so existing scripts continue to work unchanged.
"""
from __future__ import annotations

from .accounts import (
    DEFAULT_ACCOUNT_ID,
    fetch_all_accounts,
    get_margin_account_id,
    reset_margin_cache,
)
from .auth import (
    extract_access_token,
    extract_accounts_from_cookies,
    extract_identity_from_cookies,
    extract_oauth_data,
    get_session,
    is_token_expired,
    load_config,
    load_cookies,
    refresh_access_token,
    save_config,
    save_cookies,
    update_cookies_with_new_token,
)
from .client import WSClient
from .errors import (
    AuthError,
    GraphQLError,
    OrderNotFound,
    OrderRejected,
    OrderTimeout,
    WSError,
)
from .gql import graphql_query
from .models import (
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
from .orders import (
    cancel_order,
    fetch_extended_order,
    fetch_multileg_order,
    generate_order_id,
    modify_order,
    place_multileg_order,
    place_order,
    preflight_multileg,
    wait_for_multileg_order,
    wait_for_order,
)
from .positions import fetch_positions
from .queries import (
    MUTATION_ORDER_CANCEL,
    MUTATION_ORDER_CREATE,
    MUTATION_ORDER_EXECUTION_CREATE,
    MUTATION_ORDER_MODIFY,
    MUTATION_PREFLIGHT_CHECK,
    QUERY_ALL_ACCOUNTS,
    QUERY_EXTENDED_ORDER,
    QUERY_FETCH_ACTIVITIES,
    QUERY_FETCH_FINANCIALS,
    QUERY_FETCH_POSITIONS,
    QUERY_FETCH_SECURITY,
    QUERY_MULTILEG_ORDER,
    QUERY_OPTION_CHAIN,
    QUERY_OPTION_EXPIRATION_DATES,
    QUERY_SECURITY_SEARCH,
    QUERY_TRADING_BALANCE,
)
from .quotes import KNOWN_SECURITIES, resolve_symbol, search_security

__all__ = [
    # errors
    "WSError", "AuthError", "GraphQLError",
    "OrderRejected", "OrderNotFound", "OrderTimeout",
    # models
    "Order", "MultilegOrder", "Account", "Position", "Balance",
    "OrderStatus", "OrderSide", "OpenClose", "AccountType",
    # auth
    "load_config", "save_config", "load_cookies", "save_cookies",
    "extract_oauth_data", "is_token_expired",
    "refresh_access_token", "update_cookies_with_new_token",
    "extract_access_token", "extract_identity_from_cookies",
    "extract_accounts_from_cookies", "get_session",
    # gql
    "graphql_query",
    # queries
    "QUERY_FETCH_FINANCIALS", "QUERY_FETCH_POSITIONS", "QUERY_FETCH_ACTIVITIES",
    "QUERY_FETCH_SECURITY", "QUERY_SECURITY_SEARCH",
    "QUERY_OPTION_EXPIRATION_DATES", "QUERY_OPTION_CHAIN",
    "QUERY_MULTILEG_ORDER", "QUERY_EXTENDED_ORDER", "QUERY_ALL_ACCOUNTS",
    "MUTATION_ORDER_CREATE", "MUTATION_ORDER_CANCEL", "MUTATION_ORDER_MODIFY",
    "MUTATION_ORDER_EXECUTION_CREATE", "MUTATION_PREFLIGHT_CHECK",
    "QUERY_TRADING_BALANCE",
    # orders
    "generate_order_id", "place_order", "place_multileg_order",
    "cancel_order", "modify_order", "preflight_multileg",
    "fetch_multileg_order", "fetch_extended_order",
    "wait_for_order", "wait_for_multileg_order",
    # accounts
    "DEFAULT_ACCOUNT_ID", "fetch_all_accounts", "get_margin_account_id",
    "reset_margin_cache",
    # quotes
    "KNOWN_SECURITIES", "search_security", "resolve_symbol",
    # positions
    "fetch_positions",
    # client
    "WSClient",
]
