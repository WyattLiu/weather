"""Tests for ws_sdk.accounts — margin discovery with crypto filter."""
from __future__ import annotations

import pytest

from tests.conftest import load_fixture
from ws_sdk.accounts import (
    DEFAULT_ACCOUNT_ID,
    fetch_all_accounts,
    get_margin_account_id,
    reset_margin_cache,
)
from ws_sdk.models import AccountType


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_margin_cache()
    yield
    reset_margin_cache()


def test_fetch_all_accounts_returns_all_types(fake_session, patch_graphql):
    patch_graphql({"FetchAllAccounts": load_fixture("all_accounts")})
    accounts = fetch_all_accounts(fake_session, "identity-x")

    assert len(accounts) == 4
    types = [a.type for a in accounts]
    assert AccountType.CASH in types
    assert AccountType.MARGIN in types
    assert AccountType.CRYPTO in types
    assert AccountType.CREDIT_CARD in types


def test_get_margin_account_id_picks_non_crypto_non_registered(fake_session, patch_graphql):
    patch_graphql({"FetchAllAccounts": load_fixture("all_accounts")})
    margin_id = get_margin_account_id(fake_session, "identity-x")
    assert margin_id == "non-registered-BpgPfFs0QA"


def test_get_margin_account_id_caches_result(fake_session, patch_graphql):
    call_log = patch_graphql({"FetchAllAccounts": load_fixture("all_accounts")})

    first = get_margin_account_id(fake_session, "identity-x")
    second = get_margin_account_id(fake_session, "identity-x")

    assert first == second
    # Should only have hit the GraphQL layer once
    graph_calls = [c for c in call_log if c["op"] == "FetchAllAccounts"]
    assert len(graph_calls) == 1


def test_get_margin_account_id_force_refresh_busts_cache(fake_session, patch_graphql):
    call_log = patch_graphql({"FetchAllAccounts": load_fixture("all_accounts")})

    get_margin_account_id(fake_session, "identity-x")
    get_margin_account_id(fake_session, "identity-x", force_refresh=True)

    graph_calls = [c for c in call_log if c["op"] == "FetchAllAccounts"]
    assert len(graph_calls) == 2


def test_get_margin_account_id_falls_back_when_api_empty(fake_session, patch_graphql, capsys):
    patch_graphql({"FetchAllAccounts": {}})
    margin_id = get_margin_account_id(fake_session, "identity-x")
    assert margin_id == DEFAULT_ACCOUNT_ID

    out = capsys.readouterr().out
    assert "could not discover" in out.lower() or "fall" in out.lower()


def test_get_margin_account_id_skips_crypto_even_if_first(fake_session, patch_graphql):
    """If a crypto non-registered account appears before the margin one, we still pick the margin one."""
    data = {
        "identity": {
            "id": "identity-x",
            "accounts": {
                "edges": [
                    {"node": {"id": "non-registered-crypto-FIRST", "currency": "CAD"}},
                    {"node": {"id": "non-registered-REALMARGIN",   "currency": "USD"}},
                ],
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            },
        }
    }
    patch_graphql({"FetchAllAccounts": data})
    margin_id = get_margin_account_id(fake_session, "identity-x")
    assert margin_id == "non-registered-REALMARGIN"
