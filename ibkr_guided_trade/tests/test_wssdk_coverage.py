"""Coverage-focused unit tests for the ws_sdk package.

Every test runs fully offline. Network IO (``requests.post`` / ``Session.post``)
and disk IO (``~/.ws_trade``) are mocked or redirected into ``tmp_path``. The
goal is to exercise the error/refresh/transport branches that the existing
behavioural tests don't reach, pushing the SDK to ~95% line coverage.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
import requests

import ws_sdk.auth as auth
import ws_sdk.gql as gql
import ws_sdk.quotes as quotes
import ws_sdk.orders as orders
from ws_sdk.client import WSClient
from ws_sdk.errors import (
    AuthError,
    GraphQLError,
    OrderNotFound,
    OrderRejected,
    OrderTimeout,
    WSError,
)
from ws_sdk.models import OrderStatus, OrderSide, OpenClose
from tests.conftest import load_fixture


# ====================================================================== auth
# ----------------------------------------------------------------------------
# config I/O
# ----------------------------------------------------------------------------
def _redirect_config(monkeypatch, tmp_path):
    """Point the auth module's on-disk paths at a temp dir."""
    monkeypatch.setattr(auth, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(auth, "COOKIES_FILE", tmp_path / "cookies.json")
    monkeypatch.setattr(auth, "CONFIG_FILE", tmp_path / "config.json")


def test_load_config_missing(monkeypatch, tmp_path):
    _redirect_config(monkeypatch, tmp_path)
    assert auth.load_config() == {}


def test_save_and_load_config_roundtrip(monkeypatch, tmp_path):
    _redirect_config(monkeypatch, tmp_path)
    auth.save_config({"identity_id": "id-1", "device_id": "dev-9"})
    assert auth.load_config() == {"identity_id": "id-1", "device_id": "dev-9"}
    # 0600 perms applied
    mode = os.stat(tmp_path / "config.json").st_mode & 0o777
    assert mode == 0o600


# ----------------------------------------------------------------------------
# cookies atomic write + perms + crash safety
# ----------------------------------------------------------------------------
def test_save_cookies_atomic_and_perms(monkeypatch, tmp_path):
    _redirect_config(monkeypatch, tmp_path)
    auth.save_cookies({"access_token": "abc", "wssdi": "dev"})
    loaded = json.loads((tmp_path / "cookies.json").read_text())
    assert loaded == {"access_token": "abc", "wssdi": "dev"}
    mode = os.stat(tmp_path / "cookies.json").st_mode & 0o777
    assert mode == 0o600
    # No stray temp files left behind
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".cookies.")]
    assert leftover == []


def test_atomic_dump_json_failure_leaves_original_and_cleans_temp(monkeypatch, tmp_path):
    _redirect_config(monkeypatch, tmp_path)
    # Pre-existing good file
    target = tmp_path / "cookies.json"
    target.write_text(json.dumps({"original": "intact"}))

    # Force json.dump to blow up mid-write so the except branch runs.
    monkeypatch.setattr(auth.json, "dump", MagicMock(side_effect=RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        auth._atomic_dump_json({"new": "data"}, target)

    # Original file untouched (never truncated)
    assert json.loads(target.read_text()) == {"original": "intact"}
    # Temp file cleaned up
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".cookies.")]
    assert leftover == []


# ----------------------------------------------------------------------------
# refresh_access_token
# ----------------------------------------------------------------------------
def test_refresh_access_token_no_refresh_token():
    assert auth.refresh_access_token({}, "dev-1") is None


def test_refresh_access_token_success(monkeypatch):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"access_token": "new-tok", "expires_in": 1800}
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(auth.requests, "post", post)

    out = auth.refresh_access_token(
        {"refresh_token": "r1", "application_uid": "app-x"}, "dev-1"
    )
    assert out == {"access_token": "new-tok", "expires_in": 1800}
    # The refresh_token + client_id from oauth_data made it into the payload
    _, kwargs = post.call_args
    assert kwargs["json"]["refresh_token"] == "r1"
    assert kwargs["json"]["client_id"] == "app-x"
    assert kwargs["headers"]["x-ws-device-id"] == "dev-1"


def test_refresh_access_token_default_client_id(monkeypatch):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"access_token": "tok"}
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(auth.requests, "post", post)

    auth.refresh_access_token({"refresh_token": "r1"}, "dev-1")
    _, kwargs = post.call_args
    assert kwargs["json"]["client_id"] == auth.WS_CLIENT_ID


def test_refresh_access_token_non_200(monkeypatch, capsys):
    resp = MagicMock(status_code=401)
    monkeypatch.setattr(auth.requests, "post", MagicMock(return_value=resp))
    assert auth.refresh_access_token({"refresh_token": "r1"}, "dev-1") is None
    assert "Token refresh failed: 401" in capsys.readouterr().out


def test_refresh_access_token_exception(monkeypatch, capsys):
    monkeypatch.setattr(
        auth.requests, "post", MagicMock(side_effect=requests.exceptions.ConnectionError("boom"))
    )
    assert auth.refresh_access_token({"refresh_token": "r1"}, "dev-1") is None
    assert "Token refresh error" in capsys.readouterr().out


# ----------------------------------------------------------------------------
# update_cookies_with_new_token
# ----------------------------------------------------------------------------
def test_update_cookies_with_new_token(monkeypatch, tmp_path, capsys):
    _redirect_config(monkeypatch, tmp_path)
    existing_oauth = {
        "access_token": "old",
        "refresh_token": "old-r",
        "identity_canonical_id": "id-9",
    }
    cookies = {"_oauth2_access_v2": urllib.parse.quote(json.dumps(existing_oauth))}
    (tmp_path / "cookies.json").write_text(json.dumps(cookies))

    created = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    new_token = {
        "access_token": "fresh",
        "refresh_token": "fresh-r",
        "expires_in": 1800,
        "created_at": created,
    }
    out = auth.update_cookies_with_new_token(new_token)
    assert out["access_token"] == "fresh"
    assert out["refresh_token"] == "fresh-r"
    # expires_at computed from created_at + expires_in
    assert out["expires_at"].startswith("2026-01-01")
    assert "Token refreshed successfully" in capsys.readouterr().out

    # Persisted back to disk
    persisted = json.loads((tmp_path / "cookies.json").read_text())
    reparsed = json.loads(urllib.parse.unquote(persisted["_oauth2_access_v2"]))
    assert reparsed["access_token"] == "fresh"


def test_update_cookies_with_new_token_no_created_at(monkeypatch, tmp_path):
    _redirect_config(monkeypatch, tmp_path)
    cookies = {"_oauth2_access_v2": urllib.parse.quote(json.dumps({"access_token": "old"}))}
    (tmp_path / "cookies.json").write_text(json.dumps(cookies))
    # created_at absent -> no expires_at recomputed
    out = auth.update_cookies_with_new_token({"access_token": "fresh"})
    assert out["access_token"] == "fresh"
    assert "expires_at" not in out


# ----------------------------------------------------------------------------
# extract_access_token (refresh paths)
# ----------------------------------------------------------------------------
def _cookie_with(oauth: dict) -> dict:
    return {"_oauth2_access_v2": urllib.parse.quote(json.dumps(oauth))}


def test_extract_access_token_no_oauth():
    assert auth.extract_access_token({}) is None


def test_extract_access_token_valid_not_expired():
    future = (datetime.now(timezone.utc).replace(microsecond=0)).isoformat()
    # Build a clearly-future expiry
    from datetime import timedelta
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    cookies = _cookie_with({"access_token": "tok-live", "expires_at": future})
    assert auth.extract_access_token(cookies) == "tok-live"


def test_extract_access_token_expired_refresh_success(monkeypatch, tmp_path, capsys):
    _redirect_config(monkeypatch, tmp_path)
    cookies = _cookie_with({"access_token": "old", "refresh_token": "r1"})  # expired (no expires_at)
    (tmp_path / "cookies.json").write_text(json.dumps(cookies))

    monkeypatch.setattr(auth, "refresh_access_token", lambda od, dev: {"access_token": "new"})
    monkeypatch.setattr(
        auth, "update_cookies_with_new_token", lambda nt: {"access_token": "new"}
    )
    assert auth.extract_access_token(cookies) == "new"
    assert "Access token expired" in capsys.readouterr().out


def test_extract_access_token_expired_refresh_fails(monkeypatch, capsys):
    cookies = _cookie_with({"access_token": "old", "refresh_token": "r1"})
    monkeypatch.setattr(auth, "refresh_access_token", lambda od, dev: None)
    assert auth.extract_access_token(cookies) is None
    assert "Could not refresh token" in capsys.readouterr().out


# ----------------------------------------------------------------------------
# extract_accounts_from_cookies
# ----------------------------------------------------------------------------
def test_extract_accounts_from_cookies_present():
    oauth = {"profiles": {"trade": {"default": "user-123"}}}
    assert auth.extract_accounts_from_cookies(_cookie_with(oauth)) == ["user-123"]


def test_extract_accounts_from_cookies_no_default():
    oauth = {"profiles": {"trade": {}}}
    assert auth.extract_accounts_from_cookies(_cookie_with(oauth)) == []


def test_extract_accounts_from_cookies_missing():
    assert auth.extract_accounts_from_cookies({}) == []


def test_extract_accounts_from_cookies_malformed():
    assert auth.extract_accounts_from_cookies({"_oauth2_access_v2": "not-json"}) == []


# ----------------------------------------------------------------------------
# get_session
# ----------------------------------------------------------------------------
def test_get_session_no_cookies(monkeypatch, capsys):
    monkeypatch.setattr(auth, "load_cookies", lambda: {})
    monkeypatch.setattr(auth, "load_config", lambda: {})
    with pytest.raises(SystemExit):
        auth.get_session()
    assert "No cookies found." in capsys.readouterr().out


def test_get_session_no_access_token(monkeypatch, capsys):
    monkeypatch.setattr(auth, "load_cookies", lambda: {"wssdi": "dev"})
    monkeypatch.setattr(auth, "load_config", lambda: {})
    monkeypatch.setattr(auth, "extract_access_token", lambda c: None)
    with pytest.raises(SystemExit):
        auth.get_session()
    assert "No access token" in capsys.readouterr().out


def test_get_session_happy_path_saves_identity(monkeypatch):
    cookies = {"wssdi": "dev-77"}
    monkeypatch.setattr(auth, "load_cookies", lambda: cookies)
    monkeypatch.setattr(auth, "load_config", lambda: {})  # no identity_id -> triggers save
    monkeypatch.setattr(auth, "extract_access_token", lambda c: "tok-abc")
    monkeypatch.setattr(auth, "extract_identity_from_cookies", lambda c: "identity-xyz")
    saved = {}
    monkeypatch.setattr(auth, "save_config", lambda cfg: saved.update(cfg))

    sess = auth.get_session()
    assert sess.headers["Authorization"] == "Bearer tok-abc"
    assert sess.headers["x-ws-device-id"] == "dev-77"
    # identity persisted to config
    assert saved.get("identity_id") == "identity-xyz"


def test_get_session_existing_identity_no_save(monkeypatch):
    monkeypatch.setattr(auth, "load_cookies", lambda: {"wssdi": "d"})
    monkeypatch.setattr(auth, "load_config", lambda: {"identity_id": "id-pre", "device_id": "dd"})
    monkeypatch.setattr(auth, "extract_access_token", lambda c: "tok")
    called = {"saved": False}
    monkeypatch.setattr(auth, "save_config", lambda cfg: called.__setitem__("saved", True))
    sess = auth.get_session()
    assert called["saved"] is False
    assert sess.headers["Authorization"] == "Bearer tok"


# ====================================================================== gql
def _mk_session_with_response(text="{}", status=200, json_data=None, raise_exc=None):
    sess = MagicMock(spec=requests.Session)
    if raise_exc is not None:
        sess.post.side_effect = raise_exc
        return sess
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    sess.post.return_value = resp
    return sess


def test_gql_extract_operation_name_unknown():
    assert gql._extract_operation_name("{ malformed }") == "unknown"


def test_gql_three_arg_style_success():
    sess = _mk_session_with_response(text='{"data":{}}', json_data={"data": {"x": 1}})
    out = gql.graphql_query(sess, "query Foo { a }", {"v": 1})
    assert out == {"x": 1}
    _, kwargs = sess.post.call_args
    payload = kwargs["json"]
    assert payload["operationName"] == "Foo"
    assert payload["variables"] == {"v": 1}


def test_gql_four_arg_style_success():
    sess = _mk_session_with_response(text='{"data":{}}', json_data={"data": {"ok": True}})
    out = gql.graphql_query(sess, "OpName", "query Whatever { a }", {"k": "v"})
    assert out == {"ok": True}
    payload = sess.post.call_args.kwargs["json"]
    assert payload["operationName"] == "OpName"
    assert payload["variables"] == {"k": "v"}


def test_gql_request_exception(capsys):
    sess = _mk_session_with_response(raise_exc=requests.exceptions.Timeout("slow"))
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    assert "GraphQL request failed" in capsys.readouterr().out


def test_gql_non_200(capsys):
    sess = _mk_session_with_response(text="server error body", status=500)
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    out = capsys.readouterr().out
    assert "GraphQL Error: 500" in out
    assert "server error body" in out


def test_gql_empty_body(capsys):
    sess = _mk_session_with_response(text="   ", status=200)
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    assert "empty response body" in capsys.readouterr().out


def test_gql_graphql_errors_generic(capsys):
    sess = _mk_session_with_response(
        text='{"errors":[]}', json_data={"errors": [{"message": "bad field"}]}
    )
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    out = capsys.readouterr().out
    assert "GraphQL Errors" in out
    assert "bad field" in out


def test_gql_graphql_errors_unauthenticated(capsys):
    sess = _mk_session_with_response(
        text='{"errors":[]}',
        json_data={"errors": [{"message": "UNAUTHENTICATED: token gone"}]},
    )
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    assert "Session expired" in capsys.readouterr().out


def test_gql_error_without_message_key(capsys):
    # err.get('message') falls back to str(err)
    sess = _mk_session_with_response(text='{"errors":[]}', json_data={"errors": [{"x": "y"}]})
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}
    assert "GraphQL Errors" in capsys.readouterr().out


def test_gql_no_data_key_returns_empty():
    sess = _mk_session_with_response(text='{"foo":1}', json_data={"foo": 1})
    assert gql.graphql_query(sess, "query Q { a }", {}) == {}


# ====================================================================== quotes
def test_search_security_returns_results(monkeypatch):
    monkeypatch.setattr(
        quotes,
        "graphql_query",
        lambda *a, **k: {"securitySearch": {"results": [{"id": "sec-x"}]}},
    )
    out = quotes.search_security(MagicMock(), "FOO")
    assert out == [{"id": "sec-x"}]


def test_search_security_empty(monkeypatch):
    monkeypatch.setattr(quotes, "graphql_query", lambda *a, **k: {})
    assert quotes.search_security(MagicMock(), "FOO") == []


def test_search_security_none_response(monkeypatch):
    monkeypatch.setattr(quotes, "graphql_query", lambda *a, **k: None)
    assert quotes.search_security(MagicMock(), "FOO") == []


def test_resolve_symbol_known():
    assert quotes.resolve_symbol(MagicMock(), "ung") == quotes.KNOWN_SECURITIES["UNG"]


def test_resolve_symbol_via_search_and_cache(monkeypatch):
    results = [
        {"stock": {"symbol": "ZZZ"}, "status": "active", "id": "sec-zzz"},
    ]
    monkeypatch.setattr(quotes, "search_security", lambda sess, q: results)
    try:
        out = quotes.resolve_symbol(MagicMock(), "ZZZ")
        assert out == "sec-zzz"
        # cached now
        assert quotes.KNOWN_SECURITIES["ZZZ"] == "sec-zzz"
    finally:
        quotes.KNOWN_SECURITIES.pop("ZZZ", None)


def test_resolve_symbol_search_match_but_no_id(monkeypatch):
    results = [{"stock": {"symbol": "QQQ9"}, "status": "active"}]  # no id
    monkeypatch.setattr(quotes, "search_security", lambda sess, q: results)
    assert quotes.resolve_symbol(MagicMock(), "QQQ9") is None


def test_resolve_symbol_no_match(monkeypatch):
    results = [
        {"stock": {"symbol": "OTHER"}, "status": "active", "id": "sec-other"},
        {"stock": {"symbol": "NOPE"}, "status": "inactive", "id": "sec-nope"},
    ]
    monkeypatch.setattr(quotes, "search_security", lambda sess, q: results)
    assert quotes.resolve_symbol(MagicMock(), "NOPE") is None


def test_resolve_symbol_empty_results(monkeypatch):
    monkeypatch.setattr(quotes, "search_security", lambda sess, q: [])
    assert quotes.resolve_symbol(MagicMock(), "NOTHING") is None


# ====================================================================== orders
def test_place_order_shares_session_trading(monkeypatch):
    captured = {}

    def fake_q(session, op, query, variables):
        captured.update(variables["input"])
        return {"soOrdersCreateOrder": {"errors": []}}

    monkeypatch.setattr(orders, "graphql_query", fake_q)
    out = orders.place_order(
        MagicMock(), "BUY_QUANTITY", "sec-s-abc", 10, 12.5, account_id="acct-1"
    )
    assert out["order_id"].startswith("order-")
    assert captured["canonicalAccountId"] == "acct-1"
    assert captured["tradingSession"] == "ALL"  # sec-s- gets tradingSession
    assert "openClose" not in captured


def test_place_order_option_open_close(monkeypatch):
    captured = {}

    def fake_q(session, op, query, variables):
        captured.update(variables["input"])
        return {}

    monkeypatch.setattr(orders, "graphql_query", fake_q)
    orders.place_order(
        MagicMock(), "SELL_QUANTITY", "sec-o-opt", 1, 2.0, open_close="OPEN"
    )
    assert captured["openClose"] == "OPEN"
    assert "tradingSession" not in captured
    assert captured["canonicalAccountId"] == orders.DEFAULT_ACCOUNT_ID


def test_cancel_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda s, op, q, v: captured.update(v) or {"ok": 1},
    )
    out = orders.cancel_order(MagicMock(), "order-1")
    assert out == {"ok": 1}
    assert captured["cancelOrderRequest"]["externalId"] == "order-1"


def test_modify_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda s, op, q, v: captured.update(v) or {"ok": 1},
    )
    orders.modify_order(MagicMock(), "order-2", 9.99)
    assert captured["input"]["externalId"] == "order-2"
    assert captured["input"]["newLimitPrice"] == 9.99


def test_preflight_multileg(monkeypatch):
    captured = {}
    monkeypatch.setattr(orders, "graphql_query", lambda s, op, q, v: captured.update(v) or {})
    orders.preflight_multileg(
        MagicMock(),
        "acct-1",
        "act-1",
        [
            {"orderId": "o1", "side": "BUY", "securityId": "sec-o-a", "quantity": 1, "openClose": "OPEN"},
            {"orderId": "o2", "side": "SELL", "securityId": "sec-o-b", "quantity": 1},
        ],
        -1.25,
    )
    mlo = captured["input"]["activity"]["multiLegOrder"]
    assert len(mlo["orders"]) == 2
    assert mlo["orders"][0]["optionInfo"] == {"openClose": "OPEN"}
    assert "optionInfo" not in mlo["orders"][1]
    assert mlo["netCash"]["delta"] == -1.25


def test_place_multileg_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(orders, "graphql_query", lambda s, op, q, v: captured.update(v) or {})
    out = orders.place_multileg_order(
        MagicMock(),
        legs=[
            {"securityId": "sec-o-a", "orderType": "BUY_QUANTITY", "openClose": "OPEN"},
            {"securityId": "sec-o-b", "orderType": "SELL_QUANTITY"},  # default openClose OPEN
        ],
        limit_price=-0.5,
        account_id="acct-9",
    )
    assert out["order_id"].startswith("order-")
    inp = captured["input"]
    assert inp["canonicalAccountId"] == "acct-9"
    assert inp["orders"][1]["openClose"] == "OPEN"
    assert inp["limitPrice"] == -0.5


def test_fetch_multileg_order(monkeypatch):
    captured = {}
    monkeypatch.setattr(orders, "graphql_query", lambda s, op, q, v: captured.update(v) or {"x": 1})
    out = orders.fetch_multileg_order(MagicMock(), "batch-1")
    assert out == {"x": 1}
    assert captured["orderBatchId"] == "batch-1"
    assert captured["branchId"] == orders.BRANCH_ID


def test_fetch_extended_order_none(monkeypatch):
    monkeypatch.setattr(orders, "graphql_query", lambda *a, **k: {})
    assert orders.fetch_extended_order(MagicMock(), "order-x") is None


def test_fetch_extended_order_found(monkeypatch):
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda *a, **k: {"soOrdersExtendedOrder": load_fixture("extended_order_filled")},
    )
    order = orders.fetch_extended_order(MagicMock(), "order-x")
    assert order is not None
    assert order.is_terminal


def test_wait_for_order_terminal_immediately(monkeypatch):
    filled = load_fixture("extended_order_filled")
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda *a, **k: {"soOrdersExtendedOrder": filled},
    )
    order = orders.wait_for_order(MagicMock(), "order-x", now=lambda: 0.0, sleep=lambda s: None)
    assert order.is_terminal


def test_wait_for_order_times_out_with_last_state(monkeypatch):
    pending = load_fixture("extended_order_pending")
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda *a, **k: {"soOrdersExtendedOrder": pending},
    )
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(s):
        clock["t"] += 100  # jump well past the deadline

    with pytest.raises(OrderTimeout) as ei:
        orders.wait_for_order(MagicMock(), "order-x", timeout=5, now=now, sleep=sleep)
    assert ei.value.last_state is not None


def test_wait_for_order_never_found(monkeypatch):
    monkeypatch.setattr(orders, "graphql_query", lambda *a, **k: {})
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += 100

    with pytest.raises(OrderNotFound):
        orders.wait_for_order(
            MagicMock(), "order-x", timeout=5, now=lambda: clock["t"], sleep=sleep
        )


def test_wait_for_multileg_terminal(monkeypatch):
    payload = {
        "id": "batch-1",
        "status": "completed",
        "legs": [],
    }
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda *a, **k: {"soOrdersMultilegOrder": payload},
    )
    mleg = orders.wait_for_multileg_order(
        MagicMock(), "batch-1", now=lambda: 0.0, sleep=lambda s: None
    )
    assert mleg.is_terminal


def test_wait_for_multileg_times_out(monkeypatch):
    payload = {"id": "batch-1", "status": "pending", "legs": []}
    monkeypatch.setattr(
        orders, "graphql_query",
        lambda *a, **k: {"soOrdersMultilegOrder": payload},
    )
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += 100

    with pytest.raises(OrderTimeout):
        orders.wait_for_multileg_order(
            MagicMock(), "batch-1", timeout=5, now=lambda: clock["t"], sleep=sleep
        )


def test_wait_for_multileg_never_found(monkeypatch):
    monkeypatch.setattr(orders, "graphql_query", lambda *a, **k: {})
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += 100

    with pytest.raises(OrderNotFound):
        orders.wait_for_multileg_order(
            MagicMock(), "batch-1", timeout=5, now=lambda: clock["t"], sleep=sleep
        )


# ====================================================================== client
@pytest.fixture
def client(fake_session, patch_graphql):
    patch_graphql({"FetchAllAccounts": load_fixture("all_accounts")})
    with patch("ws_sdk.client.get_session", return_value=fake_session), \
         patch("ws_sdk.client.load_cookies", return_value={}), \
         patch("ws_sdk.client.extract_identity_from_cookies", return_value="identity-test"):
        return WSClient()


def test_client_init_no_identity_raises(fake_session):
    with patch("ws_sdk.client.load_cookies", return_value={}), \
         patch("ws_sdk.client.extract_identity_from_cookies", return_value=None):
        with pytest.raises(RuntimeError):
            WSClient(session=fake_session)


def test_client_init_explicit_identity_and_account(fake_session):
    ws = WSClient(session=fake_session, account_id="acct-explicit", identity_id="id-explicit")
    assert ws.identity_id == "id-explicit"
    assert ws.account_id == "acct-explicit"


def test_client_init_margin_discovery_failure_falls_back(fake_session, capsys):
    with patch("ws_sdk.client.get_margin_account_id", side_effect=RuntimeError("no margin")):
        ws = WSClient(session=fake_session, identity_id="id-x")
    from ws_sdk.accounts import DEFAULT_ACCOUNT_ID
    assert ws.account_id == DEFAULT_ACCOUNT_ID
    assert "margin discovery failed" in capsys.readouterr().out


def test_client_place_order_methods(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client._place_order",
        lambda *a, **k: {"order_id": "order-1", "result": {"soOrdersCreateOrder": {"errors": []}}},
    )
    for method, expect_side, expect_oc in [
        (client.buy_to_open, OrderSide.BUY, OpenClose.OPEN),
        (client.sell_to_open, OrderSide.SELL, OpenClose.OPEN),
        (client.buy_to_close, OrderSide.BUY, OpenClose.CLOSE),
        (client.sell_to_close, OrderSide.SELL, OpenClose.CLOSE),
    ]:
        order = method("sec-o-abc", 1, 2.5)
        assert order.side == expect_side
        assert order.open_close == expect_oc
        assert order.status == OrderStatus.PENDING


def test_client_place_passes_open_close_for_options(client, monkeypatch):
    captured = {}

    def fake_place(session, **kwargs):
        captured.update(kwargs)
        return {"order_id": "order-1", "result": {}}

    monkeypatch.setattr("ws_sdk.client._place_order", fake_place)
    client.buy_to_open("sec-o-opt", 1, 2.5)
    assert captured["open_close"] == "OPEN"
    # Equity (sec-s-) -> open_close is None
    captured.clear()
    client.buy_to_open("sec-s-eq", 1, 2.5)
    assert captured["open_close"] is None


def test_client_place_order_rejected(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client._place_order",
        lambda *a, **k: {
            "order_id": "order-bad",
            "result": {"soOrdersCreateOrder": {"errors": [{"message": "insufficient buying power", "code": "NSF"}]}},
        },
    )
    with pytest.raises(OrderRejected) as ei:
        client.buy_to_open("sec-o-abc", 1, 2.5)
    assert ei.value.code == "NSF"
    assert ei.value.external_id == "order-bad"


def test_client_place_multileg(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client._place_multileg_order",
        lambda *a, **k: {"order_id": "order-ml", "result": {}},
    )
    ml = client.place_multileg([{"securityId": "sec-o-a", "orderType": "BUY_QUANTITY", "openClose": "OPEN"}], -0.5)
    assert ml.batch_id == "order-ml"
    assert ml.status == OrderStatus.PENDING
    assert ml.limit_price == Decimal("-0.5")


def test_client_cancel_success(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client._cancel_order",
        lambda s, eid: {"orderServiceCancelOrder": {"errors": []}},
    )
    assert client.cancel("order-1") is True


def test_client_cancel_with_errors(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client._cancel_order",
        lambda s, eid: {"orderServiceCancelOrder": {"errors": [{"message": "too late"}]}},
    )
    assert client.cancel("order-1") is False


def test_client_cancel_none_result(client, monkeypatch):
    monkeypatch.setattr("ws_sdk.client._cancel_order", lambda s, eid: None)
    assert client.cancel("order-1") is True


def test_client_modify(client, monkeypatch):
    monkeypatch.setattr("ws_sdk.client._modify_order", lambda s, eid, p: {"modified": True})
    assert client.modify("order-1", 3.0) == {"modified": True}


def test_client_get_order(client, monkeypatch):
    sentinel = object()
    monkeypatch.setattr("ws_sdk.client.fetch_extended_order", lambda s, eid: sentinel)
    assert client.get_order("order-1") is sentinel


def test_client_wait_for_order(client, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "ws_sdk.client.wait_for_order",
        lambda s, eid, timeout, poll_interval: sentinel,
    )
    assert client.wait_for_order("order-1", timeout=5, poll_interval=1) is sentinel


def test_client_get_multileg_found(client, monkeypatch):
    monkeypatch.setattr(
        "ws_sdk.client.fetch_multileg_order",
        lambda s, bid: {"soOrdersMultilegOrder": {"id": "b1", "status": "pending", "legs": []}},
    )
    ml = client.get_multileg("b1")
    assert ml is not None
    assert ml.status == OrderStatus.PENDING


def test_client_get_multileg_none(client, monkeypatch):
    monkeypatch.setattr("ws_sdk.client.fetch_multileg_order", lambda s, bid: {})
    assert client.get_multileg("b1") is None


def test_client_wait_for_multileg(client, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "ws_sdk.client.wait_for_multileg_order",
        lambda s, bid, timeout, poll_interval: sentinel,
    )
    assert client.wait_for_multileg("b1", timeout=5, poll_interval=1) is sentinel


def test_client_list_accounts(client, monkeypatch):
    monkeypatch.setattr("ws_sdk.client.fetch_all_accounts", lambda s, iid: ["a", "b"])
    assert client.list_accounts() == ["a", "b"]


def test_client_list_open_orders_no_verify(client, patch_graphql):
    """verify=False path returns activity-feed stubs without extended fetch."""
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            return {
                "activityFeedItems": {
                    "edges": [
                        {"node": {
                            "accountId": "non-registered-BpgPfFs0QA",
                            "type": "DIY_BUY", "subType": "LIMIT_ORDER",
                            "unifiedStatus": "PENDING",
                            "externalCanonicalId": "order-keep",
                            "securityId": "sec-s-1", "assetQuantity": "5",
                        }},
                        # Wrong account -> filtered
                        {"node": {"accountId": "other", "type": "DIY_BUY",
                                  "externalCanonicalId": "order-wrong-acct"}},
                        # Non-tradeable type -> filtered
                        {"node": {"accountId": "non-registered-BpgPfFs0QA", "type": "CRYPTO_BUY",
                                  "externalCanonicalId": "order-crypto"}},
                        # Excluded subtype -> filtered
                        {"node": {"accountId": "non-registered-BpgPfFs0QA", "type": "DIY_BUY",
                                  "subType": "RECURRING_ORDER",
                                  "externalCanonicalId": "order-recurring"}},
                        # Terminal status -> filtered
                        {"node": {"accountId": "non-registered-BpgPfFs0QA", "type": "DIY_SELL",
                                  "unifiedStatus": "FILLED",
                                  "externalCanonicalId": "order-filled"}},
                        # No external id -> dropped
                        {"node": {"accountId": "non-registered-BpgPfFs0QA", "type": "DIY_BUY"}},
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        return {}

    patch_graphql(responder)
    stubs = client.list_open_orders(verify=False)
    ids = [o.external_id for o in stubs]
    assert ids == ["order-keep"]
    assert stubs[0].side == OrderSide.BUY
    assert stubs[0].submitted_quantity == Decimal("5")


def test_client_list_open_orders_verify_drops_terminal(client, patch_graphql):
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            return {
                "activityFeedItems": {
                    "edges": [{"node": {
                        "accountId": "non-registered-BpgPfFs0QA", "type": "DIY_BUY",
                        "unifiedStatus": "PENDING", "externalCanonicalId": "order-1",
                    }}],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        if op_name == "FetchSoOrdersExtendedOrder":
            return {"soOrdersExtendedOrder": load_fixture("extended_order_filled")}
        return {}

    patch_graphql(responder)
    assert client.list_open_orders() == []


def test_client_list_open_orders_verify_keeps_open(client, patch_graphql):
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            return {
                "activityFeedItems": {
                    "edges": [{"node": {
                        "accountId": "non-registered-BpgPfFs0QA", "type": "DIY_BUY",
                        "unifiedStatus": "PENDING", "externalCanonicalId": "order-open",
                    }}],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        if op_name == "FetchSoOrdersExtendedOrder":
            return {"soOrdersExtendedOrder": load_fixture("extended_order_pending")}
        return {}

    patch_graphql(responder)
    out = client.list_open_orders()
    assert len(out) == 1
    assert not out[0].is_terminal


def test_client_list_open_orders_verify_skips_missing(client, patch_graphql):
    """extended fetch returns None -> order skipped."""
    def responder(op_name, variables):
        if op_name == "FetchActivityFeedItems":
            return {
                "activityFeedItems": {
                    "edges": [{"node": {
                        "accountId": "non-registered-BpgPfFs0QA", "type": "DIY_BUY",
                        "unifiedStatus": "PENDING", "externalCanonicalId": "order-ghost",
                    }}],
                    "pageInfo": {"hasNextPage": False},
                }
            }
        if op_name == "FetchSoOrdersExtendedOrder":
            return {}  # not found
        return {}

    patch_graphql(responder)
    assert client.list_open_orders() == []


def test_client_list_open_orders_empty_feed(client, patch_graphql):
    patch_graphql({"FetchActivityFeedItems": {}})
    assert client.list_open_orders() == []


# ====================================================================== errors
def test_wserror_is_base():
    assert issubclass(AuthError, WSError)
    assert issubclass(GraphQLError, WSError)


def test_graphql_error_attributes():
    err = GraphQLError("boom", status_code=503, payload={"errors": []})
    assert err.status_code == 503
    assert err.payload == {"errors": []}
    assert str(err) == "boom"


def test_auth_error():
    with pytest.raises(WSError):
        raise AuthError("no cookies")


def test_order_rejected_attributes():
    err = OrderRejected("rejected", external_id="order-9", code="NSF")
    assert err.external_id == "order-9"
    assert err.code == "NSF"


def test_order_not_found_message():
    err = OrderNotFound("order-x")
    assert "order-x" in str(err)
    assert err.external_id == "order-x"


def test_order_timeout_without_last_state():
    err = OrderTimeout("order-x")
    assert err.last_state is None
    assert "order-x" in str(err)
    assert "last status" not in str(err)


def test_order_timeout_with_last_state():
    class FakeState:
        class status:
            value = "WORKING"
    err = OrderTimeout("order-x", last_state=FakeState())
    assert "WORKING" in str(err)
    assert err.last_state is not None
