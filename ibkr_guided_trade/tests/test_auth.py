"""Tests for ws_sdk.auth — cookie parsing, token expiry."""
from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone

from ws_sdk.auth import (
    extract_identity_from_cookies,
    extract_oauth_data,
    is_token_expired,
    load_cookies,
)


# ---------- is_token_expired ------------------------------------------

def test_is_token_expired_missing_expires_at():
    assert is_token_expired({}) is True
    assert is_token_expired({"access_token": "abc"}) is True


def test_is_token_expired_future_token_not_expired():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert is_token_expired({"expires_at": future}) is False


def test_is_token_expired_near_expiry_treated_as_expired():
    # 30 seconds from now — within the 60s safety window
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert is_token_expired({"expires_at": soon}) is True


def test_is_token_expired_past_token():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert is_token_expired({"expires_at": past}) is True


def test_is_token_expired_malformed_expires_at():
    assert is_token_expired({"expires_at": "not a date"}) is True


# ---------- extract_oauth_data ----------------------------------------

def test_extract_oauth_data_from_url_encoded_cookie():
    payload = {
        "access_token": "token-xyz",
        "identity_canonical_id": "identity-abc",
        "refresh_token": "refresh-xyz",
    }
    cookies = {
        "_oauth2_access_v2": urllib.parse.quote(json.dumps(payload)),
    }
    data = extract_oauth_data(cookies)
    assert data["access_token"] == "token-xyz"
    assert data["identity_canonical_id"] == "identity-abc"


def test_extract_oauth_data_missing_cookie():
    assert extract_oauth_data({}) == {}


def test_extract_oauth_data_malformed_cookie():
    assert extract_oauth_data({"_oauth2_access_v2": "not-a-json"}) == {}


def test_extract_identity_from_cookies():
    payload = {"identity_canonical_id": "identity-test-123"}
    cookies = {"_oauth2_access_v2": urllib.parse.quote(json.dumps(payload))}
    assert extract_identity_from_cookies(cookies) == "identity-test-123"


def test_extract_identity_from_empty_cookies():
    assert extract_identity_from_cookies({}) is None


# ---------- load_cookies / save_cookies round-trip -------------------

def test_load_cookies_dict_format(monkeypatch, tmp_path):
    cookies_file = tmp_path / "cookies.json"
    cookies_file.write_text(json.dumps({"access_token": "abc", "refresh_token": "xyz"}))

    monkeypatch.setattr("ws_sdk.auth.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("ws_sdk.auth.COOKIES_FILE", cookies_file)

    loaded = load_cookies()
    assert loaded == {"access_token": "abc", "refresh_token": "xyz"}


def test_load_cookies_list_format(monkeypatch, tmp_path):
    """Cookie-Editor browser extension exports a list of {name, value} dicts."""
    cookies_file = tmp_path / "cookies.json"
    cookies_file.write_text(json.dumps([
        {"name": "access_token", "value": "abc"},
        {"name": "wssdi", "value": "device-1"},
    ]))

    monkeypatch.setattr("ws_sdk.auth.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("ws_sdk.auth.COOKIES_FILE", cookies_file)

    loaded = load_cookies()
    assert loaded == {"access_token": "abc", "wssdi": "device-1"}


def test_load_cookies_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr("ws_sdk.auth.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("ws_sdk.auth.COOKIES_FILE", tmp_path / "nope.json")
    assert load_cookies() == {}
