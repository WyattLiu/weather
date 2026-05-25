#!/usr/bin/env python3
"""
Extract cookies and config from a Wealthsimple HAR file.

Usage:
    python ws_extract_cookies.py my.wealthsimple.com.har

This will:
1. Extract identity ID from OAuth token info response
2. Extract device ID from headers
3. Create ~/.ws_trade/config.json
4. Provide instructions for cookies (HAR often strips them for privacy)
"""

import json
import os
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / '.ws_trade'


def extract_from_har(har_path: str):
    print(f"Reading HAR file: {har_path}")

    with open(har_path, 'r') as f:
        har = json.load(f)

    entries = har['log']['entries']
    print(f"Found {len(entries)} entries")

    # Look for OAuth token info response
    identity_id = None
    user_id = None
    device_id = None
    email = None
    all_har_cookies = {}   # name -> value, collected from all requests

    for entry in entries:
        url = entry['request']['url']

        # Extract device ID from headers
        for h in entry['request']['headers']:
            if h['name'] == 'x-ws-device-id' and not device_id:
                device_id = h['value']

        # Collect cookies from HAR request cookies array (Safari includes these)
        for c in entry['request'].get('cookies', []):
            name = c.get('name', '')
            value = c.get('value', '')
            if name and value and name not in all_har_cookies:
                all_har_cookies[name] = value

        # Also check Cookie header directly
        for h in entry['request']['headers']:
            if h['name'].lower() == 'cookie' and h['value']:
                for pair in h['value'].split(';'):
                    pair = pair.strip()
                    if '=' in pair:
                        k, _, v = pair.partition('=')
                        k = k.strip()
                        if k and k not in all_har_cookies:
                            all_har_cookies[k] = v.strip()

        # Look for OAuth token info
        if 'oauth/v2/token/info' in url:
            resp = entry['response']
            content = resp.get('content', {})
            text = content.get('text', '')
            if text:
                try:
                    data = json.loads(text)
                    identity_id = data.get('identity_canonical_id')
                    user_id = data.get('user_canonical_id')
                    email = data.get('email')
                except:
                    pass

        # Extract access_token from OAuth token endpoint response body
        if 'oauth/v2/token' in url and 'token/info' not in url:
            text = entry['response'].get('content', {}).get('text', '')
            if text:
                try:
                    data = json.loads(text)
                    if 'access_token' in data:
                        all_har_cookies['access_token'] = data['access_token']
                    if 'refresh_token' in data:
                        all_har_cookies['refresh_token'] = data['refresh_token']
                    # Also build _oauth2_access_v2 cookie if possible
                    if data.get('access_token') and '_oauth2_access_v2' not in all_har_cookies:
                        import urllib.parse
                        all_har_cookies['_oauth2_access_v2'] = urllib.parse.quote(
                            json.dumps(data), safe=''
                        )
                except:
                    pass

    print()
    print("=" * 60)
    print("EXTRACTED VALUES")
    print("=" * 60)

    if identity_id:
        print(f"Identity ID: {identity_id}")
    else:
        print("Identity ID: NOT FOUND")
        print("  (Login to my.wealthsimple.com and capture a new HAR)")

    if user_id:
        print(f"User ID: {user_id}")

    if device_id:
        print(f"Device ID: {device_id}")

    if email:
        print(f"Email: {email}")

    # Save config
    if identity_id:
        CONFIG_DIR.mkdir(exist_ok=True)
        config = {
            'identity_id': identity_id,
            'device_id': device_id or 'cli-device-001',
            'user_id': user_id,
            'email': email,
        }
        config_file = CONFIG_DIR / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: {config_file}")

    # Save cookies if found in HAR (Safari includes them)
    TARGET_COOKIES = {'_oauth2_access_v2', 'refresh_token', 'access_token', 'device_id', 'session_id'}
    found_cookies = {k: v for k, v in all_har_cookies.items()
                     if k in TARGET_COOKIES or '_oauth' in k or 'token' in k.lower()}

    if found_cookies:
        print()
        print("=" * 60)
        print("COOKIES FOUND IN HAR")
        print("=" * 60)
        for k, v in found_cookies.items():
            print(f"  {k}: {v[:40]}...")
        CONFIG_DIR.mkdir(exist_ok=True)
        cookies_file = CONFIG_DIR / 'cookies.json'
        # Merge with any existing cookies
        existing = {}
        if cookies_file.exists():
            try:
                raw = json.loads(cookies_file.read_text())
                existing = {c['name']: c['value'] for c in raw} if isinstance(raw, list) else raw
            except:
                pass
        existing.update(found_cookies)
        cookies_file.write_text(json.dumps(existing, indent=2))
        os.chmod(cookies_file, 0o600)
        print(f"\nCookies saved to: {cookies_file}")
    else:
        print("\nNo auth cookies found in HAR (Chrome strips them).")

    # Cookie instructions
    print()
    print("=" * 60)
    print("COOKIE SETUP")
    print("=" * 60)
    print("""
The HAR export typically doesn't include cookies for privacy reasons.
You need to manually export cookies from your browser:

Option 1: Browser DevTools
  1. Open my.wealthsimple.com in Chrome
  2. F12 > Application > Cookies > my.wealthsimple.com
  3. Copy all cookie name/value pairs

Option 2: Browser Extension (recommended)
  1. Install 'EditThisCookie' or 'Cookie Editor' extension
  2. Go to my.wealthsimple.com while logged in
  3. Export cookies as JSON
  4. Save to ~/.ws_trade/cookies.json

Option 3: Console Method
  In browser DevTools console, run:
    copy(document.cookie.split('; ').map(c => {
      const [name, ...val] = c.split('=');
      return {name, value: val.join('=')};
    }))
  Then paste into ~/.ws_trade/cookies.json
""")

    cookies_file = CONFIG_DIR / 'cookies.json'
    print(f"Expected cookies file: {cookies_file}")

    # Check for any cookies in HAR
    all_cookies = set()
    for entry in entries:
        for c in entry['request'].get('cookies', []):
            if c.get('name'):
                all_cookies.add(c['name'])

    if all_cookies:
        print(f"\nCookie names found in HAR: {', '.join(sorted(all_cookies))}")
    else:
        print("\nNo cookies captured in HAR (expected - browsers strip these)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ws_extract_cookies.py <har-file>")
        print("Example: python ws_extract_cookies.py my.wealthsimple.com.har")
        sys.exit(1)

    har_path = sys.argv[1]
    if not Path(har_path).exists():
        print(f"File not found: {har_path}")
        sys.exit(1)

    extract_from_har(har_path)


if __name__ == '__main__':
    main()
