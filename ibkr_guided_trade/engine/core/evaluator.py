"""Quality evaluator — scores a state. Used to rank candidates by qΔ."""
import math
from typing import Dict
from .params import Params
from .state import PortfolioState
from .signals import z_score, regime
from .options import bs_delta, bs_gamma, bs_theta


def portfolio_greeks(state: PortfolioState) -> Dict[str, float]:
    """Compute total delta/gamma/theta from shares + options."""
    delta = float(state.shares)  # 1 delta per share
    gamma = 0.0
    theta = 0.0
    iv = max(0.01, state.iv)
    from datetime import datetime
    for opt in state.options:
        try:
            exp_d = datetime.strptime(opt.expiry, '%Y-%m-%d').date()
            dte = max(1, (exp_d - state.today).days)
        except Exception:
            continue
        T = dte / 365.0
        d = float(bs_delta(state.spot, opt.strike, T, iv, opt.right)) * 100
        g = float(bs_gamma(state.spot, opt.strike, T, iv)) * 100
        th = float(bs_theta(state.spot, opt.strike, T, iv, opt.right)) * 100
        delta += opt.qty * d
        gamma += opt.qty * g
        theta += opt.qty * th * -1  # short option theta is positive income
    return {'delta': delta, 'gamma': gamma, 'theta': theta}


def evaluate(state: PortfolioState, params: Params) -> Dict[str, float]:
    """Return scalar quality components. Sum = total quality."""
    g = portfolio_greeks(state)
    components = {}

    # Income gap (vs weekly target)
    target = params.target_weekly_income
    weekly_theta = max(0, g['theta'] * 7)
    components['income'] = -max(0, target - weekly_theta) * 1.5

    # Delta gap (target = current shares — wheel maintains share position)
    delta_gap = g['delta'] - state.shares
    components['delta_gap'] = -(delta_gap ** 2) * 0.0001

    # Gamma load (excess variance penalty)
    short_gamma_abs = abs(min(0, g['gamma']))
    T = 7 / 252.0
    var_spot = (state.iv * state.spot) ** 2 * T
    gamma_loss_full = 0.5 * short_gamma_abs * var_spot
    z = z_score(state, params)
    excess_pct = params.gamma_load_excess_pct
    if abs(z) > 0.5:
        excess_pct += params.mean_reversion_uplift
    components['gamma_load'] = -gamma_loss_full * excess_pct

    # Margin / opportunity cost on cash
    # If cash > 0: it's earning BOXX 4% (good, no penalty)
    # If cash < 0: paying margin interest (bad)
    if state.cash < 0 and params.never_negative_cash:
        # Hard penalty for negative cash
        components['negative_cash'] = state.cash * 0.06 / 52  # weekly interest cost
    else:
        components['negative_cash'] = 0.0

    # BOXX yield bonus (positive when holding BOXX)
    components['boxx_income'] = state.boxx_shares * 117 * params.boxx_yield / 52

    # Total
    components['total'] = sum(v for v in components.values())
    return components


def quality(state: PortfolioState, params: Params) -> float:
    """Single scalar quality of state."""
    return evaluate(state, params)['total']
