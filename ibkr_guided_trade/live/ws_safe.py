"""Defensive WS wrapper — retries, verifies, handles auth expiry.

WS API has historical flakiness:
  - Auth tokens silently expire mid-call
  - GraphQL rate limits return generic 500s
  - sell_to_open sometimes returns success but order doesn't appear
  - cancel sometimes 200s without actually cancelling

Strategy:
  - Every state-mutating call: retry up to N times with backoff
  - Every state-mutating call: VERIFY by querying positions/orders after
  - Auth expiry: catch + refresh once + retry
"""
from __future__ import annotations
import os
import sys
import time
from typing import Optional, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class WSError(Exception):
    pass


def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in ('unauthorized', 'token', 'expired', '401', 'authentication'))


def safe_call(fn, *args, max_retries: int = 3, backoff: float = 1.0, **kwargs) -> Any:
    """Retry wrapper with auth-refresh + exponential backoff."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if _is_auth_error(e) and attempt == 0:
                # try to refresh session once
                try:
                    from ws_sdk import refresh_session
                    refresh_session()
                    continue
                except Exception:
                    pass
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise WSError(f'failed after {max_retries} attempts: {last_err}')


def submit_and_verify(ws_client, side: str, security_id: str, qty: int,
                      price: float, verify_timeout: float = 10.0) -> dict:
    """Submit an order and verify it appears in the open-orders list.
    Returns {'external_id', 'verified': bool, 'submitted_at', ...}."""
    submit_fn = ws_client.sell_to_open if side.upper() == 'SELL_TO_OPEN' else (
                ws_client.buy_to_open if side.upper() == 'BUY_TO_OPEN' else (
                ws_client.buy_to_close if side.upper() == 'BUY_TO_CLOSE' else
                ws_client.sell_to_close))
    submitted_at = time.time()
    ord_obj = safe_call(submit_fn, security_id=security_id, qty=qty, price=price)
    ext_id = getattr(ord_obj, 'external_id', None) or getattr(ord_obj, 'id', None) or '?'
    # Verify: poll open orders for up to verify_timeout seconds
    verified = False
    deadline = time.time() + verify_timeout
    while time.time() < deadline:
        try:
            open_orders = safe_call(ws_client.list_open_orders, verify=False)
            for o in open_orders or []:
                o_ext = getattr(o, 'external_id', None) or getattr(o, 'id', None)
                if o_ext and o_ext == ext_id:
                    verified = True
                    break
            if verified:
                break
        except Exception:
            pass
        time.sleep(1)
    return {
        'external_id': ext_id,
        'verified': verified,
        'submitted_at': submitted_at,
        'side': side, 'qty': qty, 'price': price,
        'security_id': security_id,
    }


def list_open_orders_for_symbol(ws_client, security_id_prefix: Optional[str] = None,
                                  symbol_text: Optional[str] = None) -> list:
    """Filter open orders by security_id prefix or symbol text."""
    try:
        orders = safe_call(ws_client.list_open_orders, verify=False)
    except Exception:
        return []
    out = []
    for o in orders or []:
        sec = getattr(o, 'security_id', '') or ''
        sym = getattr(o, 'symbol', '') or ''
        if security_id_prefix and sec.startswith(security_id_prefix):
            out.append(o); continue
        if symbol_text and symbol_text.lower() in sym.lower():
            out.append(o); continue
    return out


def cancel_order_safe(ws_client, external_id: str) -> bool:
    """Cancel + verify it's gone from open orders."""
    try:
        safe_call(ws_client.cancel, external_id)
    except Exception as e:
        print(f'[cancel] WS cancel raised {e}; verifying state', file=sys.stderr)
    # Verify
    time.sleep(2)
    try:
        orders = safe_call(ws_client.list_open_orders, verify=False)
        for o in orders or []:
            o_ext = getattr(o, 'external_id', None) or getattr(o, 'id', None)
            if o_ext == external_id:
                return False  # still there
    except Exception:
        return False
    return True
