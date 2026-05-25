"""OAuth2 token lifecycle, cookie persistence, and session construction.

All of this code used to live at the top of ``ws_trading.py``. It is
preserved verbatim here so the behaviour (and the on-disk layout at
``~/.ws_trade/cookies.json``) stays 100% compatible with existing cookies
and downstream scripts.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ------------------------------------------------------------------ config
CONFIG_DIR = Path.home() / ".ws_trade"
COOKIES_FILE = CONFIG_DIR / "cookies.json"
CONFIG_FILE = CONFIG_DIR / "config.json"

GRAPHQL_URL = "https://my.wealthsimple.com/graphql"
TOKEN_URL = "https://api.production.wealthsimple.com/v1/oauth/v2/token"
WS_CLIENT_ID = "4da53ac2b03225bed1550eba8e4611e086c7b905a3855e6ed12ea08c246758fa"


# ------------------------------------------------------------------ config I/O
def load_config() -> dict:
    """Load saved config from ``~/.ws_trade/config.json``."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Persist config dict to ``~/.ws_trade/config.json``."""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    os.chmod(CONFIG_FILE, 0o600)


# ------------------------------------------------------------------ cookies I/O
def load_cookies() -> dict:
    """Load cookies as dict for ``requests``.

    Supports both list format (from Cookie-Editor-style exports) and dict
    format (from our own `save_cookies` writes).
    """
    if not COOKIES_FILE.exists():
        return {}
    with open(COOKIES_FILE, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {c["name"]: c["value"] for c in data}
    return data


def save_cookies(cookies: dict) -> None:
    """Persist cookies dict to ``~/.ws_trade/cookies.json`` (0600)."""
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(COOKIES_FILE, "w") as f:
        json.dump(cookies, f, indent=2)
    os.chmod(COOKIES_FILE, 0o600)


# ------------------------------------------------------------------ OAuth blob
def extract_oauth_data(cookies: dict) -> dict:
    """Extract the URL-decoded OAuth data from the ``_oauth2_access_v2`` cookie.

    Returns an empty dict if the cookie is missing or malformed.
    """
    oauth_cookie = cookies.get("_oauth2_access_v2", "")
    if not oauth_cookie:
        return {}
    decoded = urllib.parse.unquote(oauth_cookie)
    try:
        return json.loads(decoded)
    except Exception:
        return {}


def is_token_expired(oauth_data: dict) -> bool:
    """True if the access token is missing or within 60s of expiry."""
    expires_at = oauth_data.get("expires_at")
    if not expires_at:
        return True
    try:
        exp_str = expires_at.replace("Z", "+00:00")
        exp_dt = datetime.fromisoformat(exp_str)
        now = datetime.now(exp_dt.tzinfo)
        return (exp_dt - now).total_seconds() < 60
    except Exception:
        return True


def refresh_access_token(oauth_data: dict, device_id: str) -> Optional[dict]:
    """POST the refresh_token to WS OAuth and return the new token blob.

    Returns ``None`` on any error so callers can tell the user to re-export.
    """
    refresh_token = oauth_data.get("refresh_token")
    if not refresh_token:
        return None

    headers = {
        "Content-Type": "application/json",
        "x-ws-api-version": "12",
        "x-ws-device-id": device_id,
        "x-wealthsimple-client": "@wealthsimple/wealthsimple",
    }
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth_data.get("application_uid", WS_CLIENT_ID),
    }

    try:
        resp = requests.post(TOKEN_URL, json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        print(f"Token refresh failed: {resp.status_code}")
        return None
    except Exception as e:
        print(f"Token refresh error: {e}")
        return None


def update_cookies_with_new_token(new_token_data: dict) -> dict:
    """Merge a fresh OAuth payload into the on-disk cookies file."""
    cookies = load_cookies()
    oauth_data = extract_oauth_data(cookies)

    oauth_data["access_token"] = new_token_data.get("access_token", oauth_data.get("access_token"))
    oauth_data["refresh_token"] = new_token_data.get("refresh_token", oauth_data.get("refresh_token"))
    oauth_data["expires_in"] = new_token_data.get("expires_in", oauth_data.get("expires_in"))

    created_at = new_token_data.get("created_at")
    expires_in = new_token_data.get("expires_in", 1800)
    if created_at:
        expires_timestamp = created_at + expires_in
        expires_dt = datetime.fromtimestamp(expires_timestamp, tz=timezone.utc)
        oauth_data["expires_at"] = expires_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    cookies["_oauth2_access_v2"] = urllib.parse.quote(json.dumps(oauth_data))

    CONFIG_DIR.mkdir(exist_ok=True)
    with open(COOKIES_FILE, "w") as f:
        json.dump(cookies, f, indent=2)

    print("Token refreshed successfully.")
    return oauth_data


def extract_access_token(cookies: dict) -> Optional[str]:
    """Return a valid Bearer token, refreshing if expired."""
    oauth_data = extract_oauth_data(cookies)
    if not oauth_data:
        return None

    if is_token_expired(oauth_data):
        print("Access token expired, attempting refresh...")
        device_id = cookies.get("wssdi", "cli-device-001")
        new_token = refresh_access_token(oauth_data, device_id)
        if new_token:
            oauth_data = update_cookies_with_new_token(new_token)
        else:
            print("Could not refresh token. Please re-export cookies from browser.")
            return None

    return oauth_data.get("access_token")


def extract_identity_from_cookies(cookies: dict) -> Optional[str]:
    """Return the identity_canonical_id embedded in the OAuth blob."""
    oauth_data = extract_oauth_data(cookies)
    return oauth_data.get("identity_canonical_id")


def extract_accounts_from_cookies(cookies: dict) -> list:
    """Return a list of canonical trade-profile user IDs from the OAuth blob."""
    oauth_cookie = cookies.get("_oauth2_access_v2", "")
    if not oauth_cookie:
        return []

    decoded = urllib.parse.unquote(oauth_cookie)
    try:
        oauth_data = json.loads(decoded)
        profiles = oauth_data.get("profiles", {})
        trade_profile = profiles.get("trade", {})
        return [trade_profile.get("default")] if trade_profile.get("default") else []
    except Exception:
        return []


# ------------------------------------------------------------------ session
def get_session() -> requests.Session:
    """Build an authenticated :class:`requests.Session` for GraphQL calls.

    Exits the process with a friendly message if cookies are missing or the
    token cannot be refreshed.
    """
    cookies = load_cookies()
    config = load_config()

    if not cookies:
        print("No cookies found.")
        print("1. Log into https://my.wealthsimple.com")
        print("2. Open DevTools (F12) > Console")
        print(
            "3. Run: let cookies = {}; document.cookie.split(';').forEach(c => { "
            "let [k,v] = c.trim().split('='); if(k) cookies[k]=v; }); "
            "console.log(JSON.stringify(cookies, null, 2));"
        )
        print("4. Save output to ~/.ws_trade/cookies.json")
        raise SystemExit(1)

    session = requests.Session()

    access_token = extract_access_token(cookies)
    if not access_token:
        print("No access token found in cookies. Token may have expired.")
        print("Please re-export cookies from browser.")
        raise SystemExit(1)

    identity_id = config.get("identity_id") or extract_identity_from_cookies(cookies)
    if identity_id and not config.get("identity_id"):
        config["identity_id"] = identity_id
        save_config(config)

    device_id = cookies.get("wssdi", config.get("device_id", "cli-device-001"))

    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Authorization": f"Bearer {access_token}",
        "Origin": "https://my.wealthsimple.com",
        "Referer": "https://my.wealthsimple.com/app/home",
        "x-ws-api-version": "12",
        "x-ws-locale": "en-CA",
        "x-ws-profile": "trade",
        "x-platform-os": "web",
        "x-ws-device-id": device_id,
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        ),
    })
    return session
