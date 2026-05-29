"""Forward Projection Engine — week-by-week portfolio simulation.

Projects the portfolio forward 6 weeks, showing at each step:
  - Which positions expire → freed capacity
  - Income pace after expirations
  - What new trades become available
  - Distribution across 3 scenarios (down/flat/up)

The output replaces the single-number smoothness metric with a
TRAJECTORY that naturally drives DTE diversification: if week 4
shows income crashing because everything expires at 6/18, the
beam will prefer longer-DTE trades TODAY to fill the gap.

Run: python forward_projection.py --once (one-shot)
Or:  import and call project_forward() from the visualizer

Outputs to forward_cache.json — visualizer reads and incorporates
into evaluate_portfolio_quality as projected_income_stability.
"""
import json
import math
import os
import sys
import numpy as np
from datetime import date, datetime, timedelta
from scipy.stats import norm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forward_cache.json')
N_WEEKS = 6
N_PATHS = 30  # Monte Carlo paths per week


def bs_price(S, K, T, r, sigma, right='P'):
    if T <= 0.001 or sigma <= 0:
        return max(0.0, (K - S) if right == 'P' else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_theta(S, K, T, r, sigma, right='P'):
    if T <= 0.001 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -S * sigma * norm.pdf(d1) / (2 * math.sqrt(T))
    if right == 'C':
        return (term1 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 252
    return (term1 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 252


def simulate_spot(spot, iv, days):
    """Simulate spot after N days using GBM + contango."""
    dt = 1.0 / 252
    daily_vol = iv * math.sqrt(dt)
    contango = -0.0003
    price = spot
    for _ in range(days):
        z = np.random.standard_normal()
        price *= math.exp((contango - 0.5 * daily_vol**2) + daily_vol * z)
    return price


def expire_positions(positions, cutoff_date):
    """Remove positions that expire on or before cutoff_date. Returns
    (surviving_positions, expired_positions)."""
    surviving = []
    expired = []
    for pos in positions:
        exp_str, strike, right, qty, avg = pos
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        except Exception:
            surviving.append(pos)
            continue
        if exp_date <= cutoff_date:
            expired.append(pos)
        else:
            surviving.append(pos)
    return surviving, expired


def compute_weekly_theta(positions, spot, iv, ref_date):
    """Compute total daily theta from surviving positions."""
    total_theta = 0.0
    r = 0.045
    for exp_str, strike, right, qty, avg in positions:
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = max(1, (exp_date - ref_date).days)
        T = dte / 365.0
        th = bs_theta(spot, strike, T, r, iv, right)
        total_theta += abs(qty * th * 100)
    return total_theta


def bs_delta_local(S, K, T, r, sigma, right='P'):
    if T <= 0.001 or sigma <= 0:
        if right == 'C':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if right == 'C':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def compute_portfolio_delta_at_spot(positions, sim_spot, iv, ref_date, shares):
    """Re-evaluate ALL portfolio Greeks at a simulated future spot.
    Returns (total_delta, dollar_delta_per_pct, share_value_pnl)."""
    r = 0.045
    opt_delta = 0.0
    for exp_str, strike, right, qty, avg in positions:
        try:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = max(1, (exp_date - ref_date).days)
        T = dte / 365.0
        d = bs_delta_local(sim_spot, strike, T, r, iv, right) * 100
        opt_delta += qty * d
    total_delta = shares + opt_delta
    dollar_per_pct = total_delta * sim_spot * 0.01
    return total_delta, dollar_per_pct


def best_available_trade(spot, iv, ref_date, capital_free):
    """What's the best ATM put + CC at this spot? Returns combined premium."""
    best_prem = 0
    iv_adj = iv * (1.0 + max(0, (11.0 - spot) / 11.0) * 0.3)
    for dte in [14, 30, 45]:
        T = dte / 365.0
        K_put = round(spot * 2) / 2
        K_call = round(spot * 2) / 2 + 0.5
        p_prem = abs(bs_price(spot, K_put, T, 0.045, iv_adj, 'P'))
        c_prem = abs(bs_price(spot, K_call, T, 0.045, iv_adj, 'C'))
        if p_prem > 0.05:
            margin = max(1, K_put * 100 - p_prem * 100)
            n = min(5, int(capital_free / margin)) if margin > 0 else 0
            total = (p_prem + c_prem) * n * 100  # strangle
            if total > best_prem:
                best_prem = total
    return best_prem


def project_forward(positions=None, spot=None, iv=None, shares=None):
    """Run the forward projection. Returns week-by-week analysis."""
    if positions is None or spot is None:
        try:
            from ung_visualizer import OPTIONS, UNG_PRICE, SHARES
            from ung_visualizer import get_technicals_cached
            positions = list(OPTIONS)
            spot = UNG_PRICE
            shares = SHARES
            tech = get_technicals_cached() or {}
            iv = tech.get('iv', 0.45)
        except Exception as e:
            print(f"[forward] failed to load state: {e}")
            return None

    if iv is None:
        iv = 0.45
    if shares is None:
        shares = 7600

    today = date.today()
    weeks = []

    for week_num in range(1, N_WEEKS + 1):
        week_date = today + timedelta(days=7 * week_num)
        week_label = week_date.strftime('%b %d')

        # Expire positions due before this week
        surviving, expired = expire_positions(positions, week_date)

        expired_contracts = sum(abs(q) for _, _, _, q, _ in expired)
        expired_theta = compute_weekly_theta(expired, spot, iv, today) * 7

        # Surviving theta at current spot
        surviving_theta_daily = compute_weekly_theta(surviving, spot, iv, week_date)
        surviving_theta_weekly = surviving_theta_daily * 7

        # Share delta (always present)
        share_income = 0  # shares don't produce theta

        # Monte Carlo: simulate spot paths to this week
        scenarios = {'down': [], 'flat': [], 'up': []}
        all_pnls = []  # for CVaR computation
        for _ in range(N_PATHS):
            sim_spot = simulate_spot(spot, iv, 7 * week_num)
            sim_theta = compute_weekly_theta(surviving, sim_spot, iv, week_date)
            # Cycle 197: track delta evolution at this scenario
            sim_delta, sim_dollar_per_pct = compute_portfolio_delta_at_spot(
                surviving, sim_spot, iv, week_date, shares)
            # Dollar P&L = share move + option MTM change
            share_pnl = shares * (sim_spot - spot)
            # Approximate option MTM (changes per scenario)
            opt_pnl_approx = (sim_delta - shares) * (sim_spot - spot) * 0.5  # midpoint integration
            total_pnl = share_pnl + opt_pnl_approx

            freed_margin = expired_contracts * spot * 100 * 0.5
            new_trade_prem = best_available_trade(sim_spot, iv, week_date, freed_margin)

            entry = {
                'spot': round(sim_spot, 2),
                'surviving_theta_wk': round(sim_theta * 7, 0),
                'new_opportunity_prem': round(new_trade_prem, 0),
                'total_projected_wk': round(sim_theta * 7 + new_trade_prem / 4, 0),
                'sim_delta': round(sim_delta, 0),
                'dollar_pnl': round(total_pnl, 0),
                'dollar_per_pct': round(sim_dollar_per_pct, 0),
            }
            all_pnls.append(total_pnl)

            if sim_spot < spot * 0.97:
                scenarios['down'].append(entry)
            elif sim_spot > spot * 1.03:
                scenarios['up'].append(entry)
            else:
                scenarios['flat'].append(entry)

        def avg_scenario(entries, key):
            if not entries:
                return 0
            return round(np.mean([e[key] for e in entries]), 0)

        # Cycle 197: dollar P&L distribution stats
        sorted_pnls = sorted(all_pnls)
        cvar_5_pct = float(np.mean(sorted_pnls[:max(1, len(sorted_pnls) // 20)]))  # worst 5%
        cvar_25_pct = float(np.mean(sorted_pnls[:max(1, len(sorted_pnls) // 4)]))  # worst 25%
        p_loss = sum(1 for p in all_pnls if p < 0) / len(all_pnls) if all_pnls else 0
        median_pnl = float(np.median(all_pnls))
        # Delta-at-CVaR: portfolio delta in the worst-5% scenarios
        worst_5_idx = sorted(range(len(all_pnls)), key=lambda i: all_pnls[i])[:max(1, len(all_pnls)//20)]

        week_data = {
            'week': week_num,
            'date': week_date.isoformat(),
            'label': week_label,
            'expired_contracts': expired_contracts,
            'expired_theta_wk': round(expired_theta, 0),
            'surviving_contracts': sum(abs(q) for _, _, _, q, _ in surviving),
            'surviving_theta_wk': round(surviving_theta_weekly, 0),
            'cvar_5pct_pnl': round(cvar_5_pct, 0),
            'cvar_25pct_pnl': round(cvar_25_pct, 0),
            'median_pnl': round(median_pnl, 0),
            'p_loss': round(p_loss * 100, 0),
            'income_drop_pct': round(
                (1 - surviving_theta_weekly / max(1, compute_weekly_theta(positions, spot, iv, today) * 7)) * 100, 0
            ),
            'scenarios': {
                'down': {
                    'count': len(scenarios['down']),
                    'avg_theta': avg_scenario(scenarios['down'], 'surviving_theta_wk'),
                    'avg_opportunity': avg_scenario(scenarios['down'], 'new_opportunity_prem'),
                    'avg_total': avg_scenario(scenarios['down'], 'total_projected_wk'),
                },
                'flat': {
                    'count': len(scenarios['flat']),
                    'avg_theta': avg_scenario(scenarios['flat'], 'surviving_theta_wk'),
                    'avg_opportunity': avg_scenario(scenarios['flat'], 'new_opportunity_prem'),
                    'avg_total': avg_scenario(scenarios['flat'], 'total_projected_wk'),
                },
                'up': {
                    'count': len(scenarios['up']),
                    'avg_theta': avg_scenario(scenarios['up'], 'surviving_theta_wk'),
                    'avg_opportunity': avg_scenario(scenarios['up'], 'new_opportunity_prem'),
                    'avg_total': avg_scenario(scenarios['up'], 'total_projected_wk'),
                },
            },
            'rollover_urgency': (
                'CRITICAL' if expired_contracts >= 15 else
                'HIGH' if expired_contracts >= 8 else
                'MODERATE' if expired_contracts >= 3 else
                'LOW'
            ),
        }

        weeks.append(week_data)
        # Update positions for next week (carry surviving forward)
        positions = surviving

    # Compute projected income stability (replaces smoothness)
    weekly_thetas = [w['surviving_theta_wk'] for w in weeks]
    if weekly_thetas and np.mean(weekly_thetas) > 0:
        stability = max(0, 1 - np.std(weekly_thetas) / np.mean(weekly_thetas))
    else:
        stability = 0

    result = {
        'timestamp': datetime.now().isoformat(),
        'spot': spot,
        'iv': iv,
        'shares': shares,
        'n_weeks': N_WEEKS,
        'n_paths': N_PATHS,
        'weeks': weeks,
        'projected_income_stability': round(stability, 3),
        'worst_cliff_week': max(weeks, key=lambda w: w['expired_contracts'])['week'] if weeks else 0,
        'worst_cliff_contracts': max(w['expired_contracts'] for w in weeks) if weeks else 0,
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    print("[forward] running projection...")
    result = project_forward()
    if result:
        with open(CACHE_FILE, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n=== FORWARD PROJECTION ({result['n_weeks']} weeks) ===")
        print(f"Projected income stability: {result['projected_income_stability']*100:.0f}%")
        print(f"Worst cliff: week {result['worst_cliff_week']} ({result['worst_cliff_contracts']} contracts expire)")
        print()
        for w in result['weeks']:
            sc = w['scenarios']
            urgency = w['rollover_urgency']
            urg_marker = ' ⚠️' if urgency in ('HIGH', 'CRITICAL') else ''
            print(f"Week {w['week']} ({w['label']}): "
                  f"θ=${w['surviving_theta_wk']}/wk | "
                  f"-{w['expired_contracts']} expire ({urgency}{urg_marker}) | "
                  f"drop {w['income_drop_pct']}%")
            print(f"  Scenarios: "
                  f"down({sc['down']['count']}): ${sc['down']['avg_total']}/wk | "
                  f"flat({sc['flat']['count']}): ${sc['flat']['avg_total']}/wk | "
                  f"up({sc['up']['count']}): ${sc['up']['avg_total']}/wk")
        print(f"\nSaved to {CACHE_FILE}")

    if not args.once:
        import time
        while True:
            time.sleep(300)
            result = project_forward()
            if result:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"[forward] refreshed: stability={result['projected_income_stability']*100:.0f}% "
                      f"cliff=wk{result['worst_cliff_week']}({result['worst_cliff_contracts']})")


if __name__ == '__main__':
    main()
