"""Query WS for current DBA positions (shares + short puts).

Usage:
    venv/bin/python research/dba/query_positions.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ws_sdk import WSClient


def main():
    # Margin and non-margin accounts both — DBA could be in either
    client = WSClient()
    print(f'identity={client.identity_id}  account={client.account_id}\n')

    # Pull positions
    positions = client.list_positions()
    dba_holdings = []
    for p in positions:
        sym = (getattr(p, 'symbol', '') or '').upper()
        getattr(p, 'security_type', '') or ''
        # DBA shares or DBA options
        if sym.startswith('DBA') or 'DBA' in (getattr(p, 'name', '') or '').upper():
            dba_holdings.append(p)

    if not dba_holdings:
        print('No DBA positions found in margin account.')
        # Try cash account too
        print('Checking other accounts...')
        accts = client.list_accounts()
        for a in accts:
            print(f'  account {getattr(a, "id", "?")}: type={getattr(a, "type", "?")}')
        return

    print(f'DBA positions ({len(dba_holdings)}):')
    for p in dba_holdings:
        d = {k: v for k, v in vars(p).items() if not k.startswith('_')}
        print(json.dumps(d, default=str, indent=2)[:800])
        print('---')

    # Save snapshot
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'cache', 'dba_positions.json')
    with open(out, 'w') as f:
        json.dump([{k: str(v) for k, v in vars(p).items() if not k.startswith('_')}
                   for p in dba_holdings], f, indent=2)
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
