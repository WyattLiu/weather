"""Simulation executor — applies a candidate trade to a state.

Returns NEW state (immutable). Used by backtest. Live engine has its own
executor that places real orders via WS.
"""
from dataclasses import replace
from ..core.state import PortfolioState, OptionPosition


def apply(state: PortfolioState, candidate: dict) -> PortfolioState:
    """Apply a candidate trade to a state, return new state."""
    t = candidate.get('type')
    cash_delta = candidate.get('cash_delta', 0.0)
    new_cash = state.cash + cash_delta

    if t == 'OPEN_PUT':
        new_opt = OptionPosition(
            expiry=_dte_to_date(state.today, candidate['dte']),
            strike=candidate['strike'],
            right='P',
            qty=-candidate['qty'],
            avg_cost=candidate['premium'],
        )
        return replace(state, cash=new_cash, options=state.options + [new_opt])

    if t == 'COVERED_CALL':
        new_opt = OptionPosition(
            expiry=_dte_to_date(state.today, candidate['dte']),
            strike=candidate['strike'],
            right='C',
            qty=-candidate['qty'],
            avg_cost=candidate['premium'],
        )
        return replace(state, cash=new_cash, options=state.options + [new_opt])

    if t == 'BUY_PUT':
        new_opt = OptionPosition(
            expiry=_dte_to_date(state.today, candidate['dte']),
            strike=candidate['strike'],
            right='P',
            qty=candidate['qty'],
            avg_cost=candidate['cost_per_share'],
        )
        return replace(state, cash=new_cash, options=state.options + [new_opt])

    if t == 'BUY_BOXX':
        return replace(state, cash=new_cash, boxx_shares=state.boxx_shares + candidate['qty'])

    if t == 'BUY_KOLD':
        return replace(state, cash=new_cash, kold_shares=state.kold_shares + candidate['qty'])

    return state


def expire_due_options(state: PortfolioState) -> PortfolioState:
    """Process option expirations at end-of-day. Mutates positions + shares + cash."""
    spot = state.spot
    keep = []
    shares = state.shares
    cash = state.cash
    for opt in state.options:
        dte = opt.dte(state.today)
        if dte <= 0:
            # Expire / assign
            if opt.right == 'P' and opt.qty < 0 and spot < opt.strike:
                # Short put assigned → buy shares at strike
                n = abs(opt.qty) * 100
                cash -= n * opt.strike
                shares += n
            elif opt.right == 'C' and opt.qty < 0 and spot > opt.strike:
                # Short call assigned → sell shares at strike
                n = abs(opt.qty) * 100
                cash += n * opt.strike
                shares = max(0, shares - n)
            elif opt.right == 'P' and opt.qty > 0:
                # Long put — collect intrinsic
                cash += max(0, opt.strike - spot) * opt.qty * 100
            elif opt.right == 'C' and opt.qty > 0:
                cash += max(0, spot - opt.strike) * opt.qty * 100
            # Else: OTM expires worthless
        else:
            keep.append(opt)
    return replace(state, options=keep, shares=shares, cash=cash)


def _dte_to_date(today, dte: int) -> str:
    from datetime import timedelta
    return (today + timedelta(days=dte)).strftime('%Y-%m-%d')
