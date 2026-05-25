"""GraphQL transport layer for the Wealthsimple Trade SDK.

Thin wrapper around :meth:`requests.Session.post` that matches the
signature the legacy ``ws_trading.graphql_query`` used so existing
consumers don't need to change.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Union

import requests

from .auth import GRAPHQL_URL


def _extract_operation_name(query: str) -> str:
    """Return the operation name embedded in a GraphQL document."""
    m = re.match(r"(?:query|mutation|subscription)\s+(\w+)", query.strip())
    return m.group(1) if m else "unknown"


def graphql_query(
    session: requests.Session,
    query_or_op: str,
    query: Union[str, dict, None] = None,
    variables: Optional[dict] = None,
) -> dict:
    """Execute a GraphQL query or mutation.

    Supports two call styles for backward compatibility::

        graphql_query(session, QUERY_STRING, {"var": "val"})
        graphql_query(session, "OpName", QUERY_STRING, {"var": "val"})

    Returns the ``data`` field of the response, or an empty dict on
    transport errors / GraphQL errors. Errors are logged to stdout.
    """
    if query is None or isinstance(query, dict):
        # 3-arg style: query_or_op *is* the query, `query` is actually variables
        actual_query = query_or_op
        actual_variables: dict = query if isinstance(query, dict) else {}
        operation_name = _extract_operation_name(actual_query)
    else:
        operation_name = query_or_op
        actual_query = query
        actual_variables = variables or {}

    payload: dict[str, Any] = {
        "operationName": operation_name,
        "query": actual_query,
        "variables": actual_variables,
    }

    resp = session.post(GRAPHQL_URL, json=payload)

    if resp.status_code != 200:
        print(f"GraphQL Error: {resp.status_code}")
        print(resp.text[:500])
        return {}

    data = resp.json()
    if "errors" in data:
        print("GraphQL Errors:")
        for err in data["errors"]:
            msg = err.get("message", str(err))
            if "UNAUTHENTICATED" in msg or "unauthorized" in msg.lower():
                print("  Session expired. Please re-export cookies from browser.")
            else:
                print(f"  - {msg}")
        return {}

    return data.get("data", {})
