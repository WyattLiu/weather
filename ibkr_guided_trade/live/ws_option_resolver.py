"""Resolve OSI option symbols → WS security_id + live quote.

Bridges the kernel adapter (which emits OSI like 'UNG   260717P00011000')
to the WS Trading API (which needs sec-o-... security_ids).
"""
from __future__ import annotations
import os
import sys
from typing import Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)

# Module-level cache: (symbol, expiry, type) → {strike: {sec_id, bid, ask, last, oi}}
_CHAIN_CACHE: dict = {}
_CHAIN_CACHE_TS: dict = {}
import time


def parse_osi(osi: str) -> Optional[dict]:
    """UNG   260717P00011000 -> {symbol:'UNG', expiry:'2026-07-17', right:'P', strike:11.0}"""
    if not osi: return None
    parts = osi.strip().split()
    if len(parts) != 2: return None
    underlying = parts[0]
    rest = parts[1]
    if len(rest) < 15: return None
    yymmdd, right, strike_str = rest[:6], rest[6], rest[7:]
    try:
        strike = int(strike_str) / 1000
        year = 2000 + int(yymmdd[:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        expiry = f'{year:04d}-{month:02d}-{day:02d}'
        return {'symbol': underlying, 'expiry': expiry,
                'right': 'CALL' if right == 'C' else 'PUT',
                'strike': strike}
    except Exception:
        return None


def fetch_chain(symbol: str, expiry: str, right: str, cache_ttl: int = 60) -> dict:
    """Return {strike: {sec_id, bid, ask, last, oi, iv}}. Cached `cache_ttl` sec."""
    key = (symbol.upper(), expiry, right.upper())
    now = time.time()
    if key in _CHAIN_CACHE and now - _CHAIN_CACHE_TS.get(key, 0) < cache_ttl:
        return _CHAIN_CACHE[key]
    from ws_sdk import get_session, graphql_query, QUERY_OPTION_CHAIN, KNOWN_SECURITIES
    sec_id = KNOWN_SECURITIES.get(symbol.upper())
    if not sec_id:
        raise ValueError(f'Unknown underlying: {symbol}')
    session = get_session()
    # WS enum drift (June 2026): optionType now requires PUT/CALL — the
    # old single-letter P/C returns UNPROCESSABLE_ENTITY.
    opt_type = {'P': 'PUT', 'C': 'CALL'}.get(right.upper(), right.upper())
    data = graphql_query(session, 'FetchOptionChain', QUERY_OPTION_CHAIN, {
        'id': sec_id, 'expiryDate': expiry,
        'optionType': opt_type,
        'realTimeQuote': True, 'includeGreeks': True,
    })
    if not data: return {}
    chain = (data.get('security') or {}).get('optionChain') or {}
    edges = chain.get('edges') or []
    out = {}
    for edge in edges:
        node = edge.get('node') or {}
        details = node.get('optionDetails') or {}
        quote = node.get('quoteV2') or {}
        try:
            K = float(details.get('strikePrice', 0))
            if K <= 0: continue
            out[round(K, 4)] = {
                'sec_id': node.get('id', ''),
                'bid': float(quote.get('bid') or 0),
                'ask': float(quote.get('ask') or 0),
                'last': float(quote.get('last') or 0),
                'oi': int(quote.get('openInterest') or 0),
            }
        except Exception:
            continue
    _CHAIN_CACHE[key] = out
    _CHAIN_CACHE_TS[key] = now
    return out


def resolve_osi(osi: str) -> Optional[dict]:
    """OSI → {sec_id, bid, ask, last, oi, mid, strike, expiry, right}.
    Returns None if not found in the live chain.
    """
    parsed = parse_osi(osi)
    if not parsed: return None
    chain = fetch_chain(parsed['symbol'], parsed['expiry'], parsed['right'])
    leg = chain.get(round(parsed['strike'], 4))
    if not leg:
        # try matching at 2 decimal places
        for k, v in chain.items():
            if abs(k - parsed['strike']) < 0.005:
                leg = v; break
    if not leg:
        return None
    mid = (leg['bid'] + leg['ask']) / 2 if leg['bid'] > 0 and leg['ask'] > 0 else (leg['last'] or 0)
    return {**parsed, **leg, 'mid': round(mid, 3)}


if __name__ == '__main__':
    import json
    if len(sys.argv) > 1:
        for osi in sys.argv[1:]:
            r = resolve_osi(osi)
            print(json.dumps(r, indent=2, default=str))
    else:
        # Test
        for osi in ['UNG   260717P00011000', 'UNG   260717P00012000']:
            r = resolve_osi(osi)
            print(f'{osi} → sec_id={r["sec_id"] if r else None} bid=${r["bid"] if r else None} ask=${r["ask"] if r else None}')
