"""Multi-objective quality scorer for trade candidates (item #3 port).

CRITICAL approach: production's evaluate_portfolio_quality has
hand-tuned weights across many cycles (income_gap × 1.5 asymmetric,
pillar_drift cap 0.0006, smoothness +$500, etc.) — arbitrary by
construction. This port builds the SCAFFOLDING with explicit tunable
weights so the backtest can EMPIRICALLY derive what matters.

Components (all $-normalized):
  income_score    = (current weekly theta - target) * income_weight
                    asymmetric: shortfall costs more than overshoot
  dd_penalty      = -projected_30d_drawdown * dd_weight (negative)
  delta_gap       = -((delta - target_delta) ** 2) * delta_gap_weight
  smoothness_bonus = (cv_inverse - 0.5) * smoothness_weight
  tail_hedge_score = (n_long_puts - floor) * tail_weight  (penalty if below)

Returns a SINGLE scalar (higher = better). Use in beam search to compare
candidate trades.
"""
import statistics


def score_portfolio_quality(
    state: dict,
    weights: dict | None = None,
) -> dict:
    """Score a portfolio state with multi-objective metrics.

    Args:
        state: dict with current portfolio + market context
          required keys: weekly_theta, target_weekly_income,
                         total_delta, target_delta, max_drawdown,
                         n_long_puts, tail_floor, recent_premium
        weights: dict overrides for component weights

    Returns:
        dict with 'total' score + component breakdown
    """
    w = {
        'income': 1.0,
        'income_shortfall_mult': 1.5,
        'dd': 0.5,
        'delta_gap': 0.001,
        'smoothness': 200.0,
        'tail_hedge': 500.0,
    }
    if weights:
        w.update(weights)

    weekly_theta = float(state.get('weekly_theta', 0))
    target = float(state.get('target_weekly_income', 1500))
    gap = weekly_theta - target
    income_score = gap * w['income'] if gap >= 0 else gap * w['income'] * w['income_shortfall_mult']

    dd = float(state.get('max_drawdown', 0))  # negative number
    dd_penalty = dd * w['dd']  # already negative

    delta = float(state.get('total_delta', 0))
    target_delta = float(state.get('target_delta', 0))
    delta_gap_score = -((delta - target_delta) ** 2) * w['delta_gap']

    recent = state.get('recent_premium', [])
    smoothness_bonus = 0.0
    if recent and len(recent) >= 4:
        m = sum(recent) / len(recent)
        if m > 0:
            std = statistics.pstdev(recent)
            cv = std / m
            smoothness_bonus = (0.5 - cv) * w['smoothness']

    n_long = int(state.get('n_long_puts', 0))
    floor = int(state.get('tail_floor', 2))
    tail_score = (n_long - floor) * w['tail_hedge'] if n_long < floor else 0

    total = income_score + dd_penalty + delta_gap_score + smoothness_bonus + tail_score

    return {
        'total': total,
        'income': income_score,
        'dd': dd_penalty,
        'delta_gap': delta_gap_score,
        'smoothness': smoothness_bonus,
        'tail_hedge': tail_score,
    }


def score_candidate_delta(before_state: dict, after_state: dict,
                          weights: dict | None = None) -> float:
    """Compute the quality DELTA of applying a candidate trade.

    Production: emit candidates, apply each to a state copy, evaluate
    quality of both states, score = q_after - q_before. Best delta wins.
    """
    q_before = score_portfolio_quality(before_state, weights)['total']
    q_after = score_portfolio_quality(after_state, weights)['total']
    return q_after - q_before


if __name__ == '__main__':
    # Sanity test
    state = {
        'weekly_theta': 800,
        'target_weekly_income': 1500,
        'total_delta': 5000,
        'target_delta': 6000,
        'max_drawdown': -10000,
        'n_long_puts': 1,
        'tail_floor': 2,
        'recent_premium': [800, 900, 700, 800],
    }
    q = score_portfolio_quality(state)
    print('=== Sample portfolio quality ===')
    for k, v in q.items():
        print(f'  {k:<12} ${v:>+10,.0f}')
