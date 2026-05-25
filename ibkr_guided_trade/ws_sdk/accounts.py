"""Account discovery and the "margin only" filter.

The SDK only ever trades equity/options on the Wealthsimple margin
account. Crypto, cash savings, and credit-card accounts are all
classified by :class:`AccountType` and ignored.

:func:`get_margin_account_id` walks the account list returned by the
``FetchAllAccounts`` query and returns the first non-crypto
non-registered ID. The result is cached in-process; callers can bust
the cache with :func:`reset_margin_cache`.
"""
from __future__ import annotations

from typing import List, Optional

import requests

from .gql import graphql_query
from .models import Account
from .queries import QUERY_ALL_ACCOUNTS

# Fallback constant — historically hard-coded in ws_trading.py. Kept as
# the last-resort value if the FetchAllAccounts query ever fails.
DEFAULT_ACCOUNT_ID: str = "non-registered-BpgPfFs0QA"

_margin_cache: Optional[str] = None


def reset_margin_cache() -> None:
    """Clear the cached margin account ID (mostly for tests)."""
    global _margin_cache
    _margin_cache = None


def fetch_all_accounts(
    session: requests.Session,
    identity_id: str,
    page_size: int = 25,
) -> List[Account]:
    """Return all accounts on this identity, normalised to :class:`Account`."""
    data = graphql_query(
        session,
        "FetchAllAccounts",
        QUERY_ALL_ACCOUNTS,
        {"identityId": identity_id, "pageSize": page_size, "cursor": None},
    )
    if not data:
        return []

    identity = data.get("identity") or {}
    accounts = (identity.get("accounts") or {}).get("edges") or []
    return [Account.from_node(edge.get("node") or {}) for edge in accounts if edge]


def get_margin_account_id(
    session: requests.Session,
    identity_id: str,
    force_refresh: bool = False,
) -> str:
    """Return the canonical margin (non-registered, non-crypto) account ID.

    Cached in-process after the first successful lookup. If the API
    returns zero margin accounts (unexpected), falls back to the legacy
    :data:`DEFAULT_ACCOUNT_ID` constant so long-running scripts don't
    hard-crash.
    """
    global _margin_cache
    cached = _margin_cache
    if cached and not force_refresh:
        return cached

    try:
        accounts = fetch_all_accounts(session, identity_id)
    except Exception as exc:  # pragma: no cover — defensive
        print(f"[ws_sdk.accounts] FetchAllAccounts failed: {exc}")
        accounts = []

    for acct in accounts:
        if acct.is_margin:
            _margin_cache = acct.id
            return acct.id

    # Fallback: no margin account discovered. Warn and return the legacy ID.
    print(
        "[ws_sdk.accounts] WARN: could not discover margin account via API, "
        f"falling back to {DEFAULT_ACCOUNT_ID}"
    )
    _margin_cache = DEFAULT_ACCOUNT_ID
    return DEFAULT_ACCOUNT_ID
