"""Shared pytest fixtures for the ws_sdk test suite.

All tests run fully offline — :func:`fake_session` + :func:`patch_graphql`
intercept every GraphQL call and dispatch to canned JSON fixtures from
``tests/fixtures/``. No cookies, no network, no ``~/.ws_trade``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import MagicMock

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file by basename (without .json)."""
    path = FIXTURE_DIR / f"{name}.json"
    return json.loads(path.read_text())


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def fake_session():
    """A requests.Session stand-in that no test should actually call."""
    return MagicMock(name="fake_session")


@pytest.fixture
def patch_graphql(monkeypatch):
    """Return a helper that patches ``ws_sdk.gql.graphql_query``.

    Example::

        def test_thing(patch_graphql):
            patch_graphql({"FetchSoOrdersExtendedOrder": load_fixture("extended_order_filled")})
            ...
    """
    calls: list[dict] = []

    def _install(responses: Dict[str, Any] | Callable[[str, dict], Any]):
        def fake_query(session, query_or_op, query=None, variables=None, **kwargs):
            # Normalise: support both 3-arg and 4-arg call styles
            if query is None or isinstance(query, dict):
                op_name = _guess_op(query_or_op)
                vars_ = query if isinstance(query, dict) else {}
            else:
                op_name = query_or_op
                vars_ = variables or {}
            calls.append({"op": op_name, "vars": vars_})

            if callable(responses):
                return responses(op_name, vars_)
            if op_name in responses:
                resp = responses[op_name]
                return resp(vars_) if callable(resp) else resp
            return {}

        # Patch all the places graphql_query is imported from.
        for modpath in (
            "ws_sdk.gql",
            "ws_sdk.orders",
            "ws_sdk.accounts",
            "ws_sdk.positions",
            "ws_sdk.client",
            "ws_sdk.quotes",
        ):
            monkeypatch.setattr(f"{modpath}.graphql_query", fake_query, raising=False)
        return calls

    return _install


def _guess_op(query_str: str) -> str:
    import re
    m = re.match(r"(?:query|mutation)\s+(\w+)", (query_str or "").strip())
    return m.group(1) if m else "unknown"


@pytest.fixture
def fake_clock():
    """A deterministic (now, sleep) pair for wait_for_order tests."""
    class Clock:
        def __init__(self):
            self.t = 0.0
            self.sleeps: list[float] = []

        def now(self) -> float:
            return self.t

        def sleep(self, s: float) -> None:
            self.sleeps.append(s)
            self.t += s

    return Clock()
