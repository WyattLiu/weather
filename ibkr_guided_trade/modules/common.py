"""
Common utilities shared across all modules.
"""

from ib_insync import IB
from datetime import datetime
import pytz
import os

# Connection settings
TRADING_CLIENT_ID = 50
IBKR_HOST = '192.168.1.127'
IBKR_PORT = 20009
DEFAULT_ACCOUNT = 'U10366498'

# Timezone
ET = pytz.timezone('US/Eastern')

# Database path (for algo module)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies.db')


def connect(client_id=None):
    """Connect to IBKR with fixed client ID"""
    ib = IB()
    cid = client_id if client_id is not None else TRADING_CLIENT_ID
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=cid, timeout=30)
    return ib


def get_timestamp():
    """Get current timestamp in Eastern time"""
    return datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')


def format_currency(value, width=12):
    """Format a number as currency with alignment"""
    if value is None:
        return "N/A".rjust(width)
    return f"${float(value):>,.2f}".rjust(width)
