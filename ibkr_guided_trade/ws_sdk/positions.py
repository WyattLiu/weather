"""Position fetching with the built-in margin-only filter.

:func:`fetch_positions` wraps the ``FetchIdentityPositions`` GraphQL
query and returns a list of typed :class:`Position` objects. By default
it restricts the query to the margin account ID so crypto holdings
never pollute the output.
"""
from __future__ import annotations

from typing import List, Optional

import requests

from .gql import graphql_query
from .models import Position
from .queries import QUERY_FETCH_POSITIONS


def fetch_positions(
    session: requests.Session,
    identity_id: str,
    account_ids: Optional[List[str]] = None,
    currency: str = "USD",
    first: int = 100,
    aggregated: bool = True,
) -> List[Position]:
    """Fetch positions for the given identity, filtered to ``account_ids``.

    If ``account_ids`` is ``None`` (default) the query returns all
    accounts — callers that want margin-only should pass
    ``[margin_account_id]`` explicitly. :meth:`WSClient.list_positions`
    handles that automatically.
    """
    variables = {
        "identityId": identity_id,
        "currency": currency,
        "first": first,
        "aggregated": aggregated,
        "currencyOverride": "MARKET",
        "sort": "TODAY_GAIN",
        "includeSecurity": True,
        "includeAccountData": True,
        "includeOneDayReturnsBaseline": True,
    }
    if account_ids:
        variables["accountIds"] = account_ids

    data = graphql_query(
        session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, variables
    )
    if not data:
        return []

    edges = (
        (data.get("identity") or {})
        .get("financials", {})
        .get("current", {})
        .get("positions", {})
        .get("edges", [])
    ) or []

    return [Position.from_position_v2(edge.get("node") or {}) for edge in edges if edge]
