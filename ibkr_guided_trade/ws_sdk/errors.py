"""Exceptions raised by the Wealthsimple Trade SDK."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Order


class WSError(Exception):
    """Base class for all ws_sdk errors."""


class AuthError(WSError):
    """Raised when cookies/tokens are missing, invalid, or cannot be refreshed."""


class GraphQLError(WSError):
    """Raised when the GraphQL endpoint returns a non-2xx or an errors payload."""

    def __init__(self, message: str, status_code: Optional[int] = None, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class OrderRejected(WSError):
    """Raised when the order mutation returned a rejection."""

    def __init__(self, message: str, external_id: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.external_id = external_id
        self.code = code


class OrderNotFound(WSError):
    """Raised when an external_id cannot be located after placement."""

    def __init__(self, external_id: str):
        super().__init__(f"Order not found: {external_id}")
        self.external_id = external_id


class OrderTimeout(WSError):
    """Raised when wait_for_order exceeds its deadline with the order still working.

    The last known Order snapshot is attached as ``last_state`` so callers
    can decide to cancel, reprice, or keep waiting without re-polling.
    """

    def __init__(self, external_id: str, last_state: "Optional[Order]" = None):
        msg = f"Order timed out: {external_id}"
        if last_state is not None:
            msg += f" (last status: {last_state.status.value})"
        super().__init__(msg)
        self.external_id = external_id
        self.last_state = last_state
