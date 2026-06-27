"""What-If Refiner — Monte Carlo Tree Search for options portfolio.

Runs continuously in the background (like AlphaGo's pondering).
Simulates UNG spot paths, evaluates best trades at each future state,
propagates expected values back to today's decision.

Results cached to whatif_cache.json — the visualizer reads this during
evaluate_portfolio_quality to incorporate forward-looking opportunity
cost into recommendations.

Run: python whatif_refiner.py (background daemon)
Or: scheduled via cron every 5 minutes
"""
import json
import time
import math
import os
import sys
import numpy as np
from datetime import date, datetime
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'whatif_cache.json')
N_SCENARIOS = 5       # spot moves to simulate
N_ITERATIONS = 50     # rollouts per cycle
HORIZON_DAYS = 7      # lookahead horizon


def bs_price(S, K, T, r, sigma, right='P'):
    if T <= 0.001 or sigma <= 0:
        return max(0.0, (K - S) if right == 'P' else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_current_state():
    """Load current portfolio state from the visualizer's live data."""
    try:
        from ung_visualizer import (
            OPTIONS, UNG_PRICE, SHARES,
            compute_portfolio_state, get_technicals_cached,
        )
        spot = UNG_PRICE
        tech = get_technicals_cached() or {}
        iv = tech.get('iv', 0.45)
        ps = compute_portfolio_state(OPTIONS, spot, iv, date.today())
        return {
            'spot': spot,
            'iv': iv,
            'shares': SHARES,
            'options': list(OPTIONS),
            'total_delta': ps.get('total_delta', 0),
            'total_gamma': ps.get('total_gamma', 0),
            'total_theta': ps.get('total_theta', 0),
            'positions': ps.get('positions', []),
        }
    except Exception as e:
        print(f"[whatif] failed to load state: {e}")
        return None


def simulate_spot_paths(spot, iv, horizon_days, n_paths):
    """Generate simulated UNG spot paths using geometric Brownian motion."""
    dt = 1.0 / 252
    daily_vol = iv * math.sqrt(dt)
    # UNG-specific: contango drag ~0.03%/day
    contango_drift = -0.0003

    paths = []
    for _ in range(n_paths):
        price = spot
        path = [price]
        for _ in range(horizon_days):
            z = np.random.standard_normal()
            price *= math.exp((contango_drift - 0.5 * daily_vol**2) + daily_vol * z)
            path.append(price)
        paths.append(path)
    return paths


def evaluate_opportunities_at_spot(sim_spot, iv, capital_available):
    """At a given spot, evaluate the BEST trade across multiple strikes/DTEs.

    Scans ATM, 5%OTM, 10%OTM puts AND ATM covered calls at 14/30/45 DTE.
    Returns the single best trade by premium/margin efficiency — this is
    what the operator WOULD do at this simulated spot.
    """
    iv_adj = iv * (1.0 + max(0, (11.0 - sim_spot) / 11.0) * 0.3)
    best = None

    for dte in [14, 30, 45]:
        T = dte / 365.0
        for otm_pct in [0.0, 0.05, 0.10]:
            K = round((sim_spot * (1 - otm_pct)) * 2) / 2
            if K <= 0:
                continue
            prem = abs(bs_price(sim_spot, K, T, 0.045, iv_adj, 'P'))
            if prem < 0.05:
                continue
            margin_per = max(1, K * 100 - prem * 100)
            n = min(10, int(capital_available / margin_per)) if margin_per > 0 else 0
            if n <= 0:
                continue
            total = prem * n * 100
            efficiency = total / (margin_per * n) if margin_per * n > 0 else 0
            entry = {
                'spot': round(sim_spot, 2),
                'strike': K,
                'dte': dte,
                'otm_pct': round(otm_pct * 100, 0),
                'premium_per_share': round(prem, 3),
                'contracts': n,
                'total_premium': round(total, 0),
                'efficiency': round(efficiency * 100, 1),
                'iv_adj': round(iv_adj, 3),
            }
            if best is None or total > best['total_premium']:
                best = entry

        # Also check covered calls (ATM + 5%OTM)
        for otm_pct in [0.0, 0.05]:
            K_c = round((sim_spot * (1 + otm_pct)) * 2) / 2
            prem_c = abs(bs_price(sim_spot, K_c, T, 0.045, iv_adj, 'C'))
            if prem_c < 0.05:
                continue
            total_c = prem_c * 5 * 100  # assume 5 covered calls
            entry_c = {
                'spot': round(sim_spot, 2),
                'strike': K_c,
                'dte': dte,
                'type': 'CC',
                'otm_pct': round(otm_pct * 100, 0),
                'premium_per_share': round(prem_c, 3),
                'contracts': 5,
                'total_premium': round(total_c, 0),
                'efficiency': 99.9,  # covered calls use shares, no margin
                'iv_adj': round(iv_adj, 3),
            }
            if best is None or total_c > best['total_premium']:
                best = entry_c

    return best or {
        'spot': round(sim_spot, 2), 'strike': 0, 'premium_per_share': 0,
        'contracts': 0, 'total_premium': 0, 'iv_adj': round(iv_adj, 3),
    }


def run_mcts_iteration(state):
    """One MCTS iteration: simulate paths, evaluate opportunities, backprop."""
    spot = state['spot']
    iv = state['iv']

    # Estimate current capital and margin usage
    total_margin_used = sum(
        max(0, K * abs(qty) * 100 - (avg / 100 if avg > 1 else avg) * abs(qty) * 100)
        for exp, K, right, qty, avg in state['options']
        if qty < 0 and right == 'P'
    )
    capital = 112000  # approximate
    free_margin = max(0, capital - total_margin_used)

    # Simulate spot paths
    paths = simulate_spot_paths(spot, iv, HORIZON_DAYS, N_ITERATIONS)

    # At each path's endpoint, evaluate what opportunities exist
    scenario_results = []
    for path in paths:
        end_spot = path[-1]
        opp = evaluate_opportunities_at_spot(end_spot, iv, free_margin)
        scenario_results.append(opp)

    # Aggregate: expected premium available after holding margin
    avg_premium = np.mean([r['total_premium'] for r in scenario_results])
    # Compare to deploying NOW at current spot
    current_opp = evaluate_opportunities_at_spot(spot, iv, free_margin)

    # Opportunity value = expected future premium - current premium
    # If future premium > current (because vol might spike or spot might drop),
    # there's value in WAITING (keeping margin free)
    opportunity_value = avg_premium - current_opp['total_premium']

    # Scenarios breakdown by spot direction
    down_scenarios = [r for r in scenario_results if r['spot'] < spot * 0.97]
    up_scenarios = [r for r in scenario_results if r['spot'] > spot * 1.03]
    flat_scenarios = [r for r in scenario_results if spot * 0.97 <= r['spot'] <= spot * 1.03]

    return {
        'timestamp': datetime.now().isoformat(),
        'spot': spot,
        'iv': iv,
        'free_margin': round(free_margin, 0),
        'n_simulations': N_ITERATIONS,
        'horizon_days': HORIZON_DAYS,
        'current_best_premium': current_opp['total_premium'],
        'expected_future_premium': round(avg_premium, 0),
        'opportunity_value': round(opportunity_value, 0),
        'scenarios': {
            'down': {
                'count': len(down_scenarios),
                'avg_premium': round(np.mean([r['total_premium'] for r in down_scenarios]), 0) if down_scenarios else 0,
                'avg_spot': round(np.mean([r['spot'] for r in down_scenarios]), 2) if down_scenarios else 0,
            },
            'flat': {
                'count': len(flat_scenarios),
                'avg_premium': round(np.mean([r['total_premium'] for r in flat_scenarios]), 0) if flat_scenarios else 0,
            },
            'up': {
                'count': len(up_scenarios),
                'avg_premium': round(np.mean([r['total_premium'] for r in up_scenarios]), 0) if up_scenarios else 0,
                'avg_spot': round(np.mean([r['spot'] for r in up_scenarios]), 2) if up_scenarios else 0,
            },
        },
        'recommendation': (
            'DEPLOY_NOW' if opportunity_value < -50 else
            'HOLD_MARGIN' if opportunity_value > 100 else
            'NEUTRAL'
        ),
    }


def refine_loop(single_run=False):
    """Main refinement loop. Runs continuously or once."""
    print(f"[whatif] refiner started (n_iter={N_ITERATIONS}, horizon={HORIZON_DAYS}d)")

    while True:
        state = get_current_state()
        if state is None:
            print("[whatif] no state available, sleeping 60s")
            time.sleep(60)
            if single_run:
                break
            continue

        result = run_mcts_iteration(state)

        # Save to cache file
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"[whatif] cycle complete: opp_value=${result['opportunity_value']:+.0f} "
                  f"({result['recommendation']}) | "
                  f"down={result['scenarios']['down']['count']} "
                  f"flat={result['scenarios']['flat']['count']} "
                  f"up={result['scenarios']['up']['count']}")
        except Exception as e:
            print(f"[whatif] save failed: {e}")

        if single_run:
            break

        time.sleep(300)  # 5 minutes between refinements


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run once then exit')
    args = parser.parse_args()
    refine_loop(single_run=args.once)
