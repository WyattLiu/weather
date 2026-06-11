"""Static security catalog and quote helpers.

``KNOWN_SECURITIES`` is a hand-maintained map of ticker symbol to
Wealthsimple canonical security ID. These IDs were harvested from HAR
captures; if Wealthsimple ever re-issues an ID (e.g. post-split) the
value here needs updating. :func:`search_security` provides a dynamic
lookup via the GraphQL ``securitySearch`` endpoint.
"""
from __future__ import annotations

from typing import Optional

import requests

from .gql import graphql_query
from .queries import QUERY_SECURITY_SEARCH

KNOWN_SECURITIES: dict[str, str] = {
    "UNG":  "sec-s-32f0b46791214cbcbee9486e40232ea4",
    "DBA":  "sec-s-28b77b54d97d425baf17180097a088e5",
    "CORN": "sec-s-ecca4d4b877a4b35a81790f17e78d27f",
    "WEAT": "sec-s-9140916198574c2cb11d0224f7e76626",
    "SOYB": "sec-s-8c0cbc4cdc724492b7fd51e399ef69a0",
    "CANE": "sec-s-2cdecbb2bd544e65845b76cbcdb88692",
    "KOLD": "sec-s-c8f82308d6684084b0a484981d81b03f",
    "HUBB": "sec-s-acba8c87de6a4dc9ad1d2f478aeacd17",
    "ETN":  "sec-s-f655317dc6954c3d900001b77ef7e6fe",
    "INTC": "sec-s-4be2419c19f1402c8b74e5af5111003e",
    "AMD":  "sec-s-412f6aa6c06d4c13a1e020534bdd84e4",
    "STX":  "sec-s-c0e8d6a5130c4c19aab62a4cdc0a8a6d",
    "SPY":  "sec-s-27167ecbd81140fe9cdc02535f43174d",
    "NVDA": "sec-s-220e8c65080c441aa87da8089460fae4",
}


def search_security(session: requests.Session, query: str) -> list[dict]:
    """Search for a security by ticker or company name.

    Returns the raw ``securitySearch.results`` list from the GraphQL
    response, or an empty list on error.
    """
    data = graphql_query(
        session,
        "FetchSecuritySearchResult",
        QUERY_SECURITY_SEARCH,
        {"query": query, "securityGroupIds": None},
    )
    return (data or {}).get("securitySearch", {}).get("results", []) or []


def resolve_symbol(session: requests.Session, symbol: str) -> Optional[str]:
    """Return the canonical security ID for ``symbol``.

    Checks :data:`KNOWN_SECURITIES` first, falls back to a live search
    and caches the result in the in-memory dict.
    """
    sym = symbol.upper().strip()
    if sym in KNOWN_SECURITIES:
        return KNOWN_SECURITIES[sym]

    results = search_security(session, sym)
    for r in results:
        stock = r.get("stock") or {}
        if stock.get("symbol", "").upper() == sym and r.get("status") == "active":
            sec_id = r.get("id")
            if sec_id:
                KNOWN_SECURITIES[sym] = sec_id
                return sec_id
    return None
