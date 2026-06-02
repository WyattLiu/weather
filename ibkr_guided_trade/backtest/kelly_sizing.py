"""Kelly position sizing for backtest engine.

Simplified port of production's compute_kelly. Production uses
ScenarioDistribution (multi-horizon quantile kernel with seasonal_drift +
supply_regime + pillars); this module uses BS analytics directly.

Kelly: f* = (p_win * W - p_loss * L) / L
  where W = premium kept if OTM, L = loss if assigned
"""
import math
from scipy.stats import norm


def kelly_qty_short_put(spot: float, strike: float, dte: int, iv: float,
                       cash_available: float, premium: float,
                       max_frac: float = 0.25,
                       kelly_safety: float = 0.5,
                       model_conviction: float = 0.0,
                       scenario_dist=None) -> int:
    """Kelly-sized qty for a short cash-secured put.

    Args:
        spot: current UNG price
        strike: put strike
        dte: days to expiry
        iv: implied vol (annualized, e.g., 0.55)
        cash_available: free cash for collateral
        premium: per-share BS premium
        max_frac: cap fraction of cash (production uses 0.25)
        kelly_safety: half-kelly (0.5) is standard safety derate

    Returns:
        int qty (number of contracts), >= 1 if any room
    """
    if dte <= 0 or iv <= 0 or strike <= 0 or premium <= 0 or cash_available <= 0:
        return 0

    T = dte / 365.0
    # P(spot_T < strike) — prefer ScenarioDistribution if provided
    # (proper 7-point quantile kernel with seasonal_drift + contango).
    # Fall back to BS d2 (lognormal, zero drift) otherwise.
    if scenario_dist is not None:
        p_otm_bs = float(scenario_dist.prob_above(strike, dte))
    else:
        d2 = (math.log(spot / strike) - 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
        p_otm_bs = float(norm.cdf(d2))   # prob spot > strike per BS

    # CONVICTION ADJUSTMENT (the "firmness" the user asked for):
    # BS assumes zero-drift random walk. In reality, when model conviction
    # is high (storage Z cheap + momentum bouncing + seasonality favorable),
    # the drift IS bullish for NG → puts more likely to expire OTM than BS
    # predicts. model_conviction ∈ [-0.20, +0.20] adds directly to p_otm.
    # Positive = bullish drift (sell more puts), negative = bearish (less).
    p_otm = max(0.01, min(0.99, p_otm_bs + model_conviction))

    if p_otm < 0.5:
        return 0  # ITM puts (or strongly bearish model) — skip
    conviction = (p_otm - 0.5) * 2.0   # 0 to 1
    f = max(0.0, min(conviction * kelly_safety, max_frac))

    # Convert fraction → contracts (collateral = strike * 100 per contract)
    coll_per_contract = strike * 100
    if coll_per_contract <= 0:
        return 0
    max_capital = cash_available * f
    qty = int(max_capital // coll_per_contract)
    return max(0, min(qty, 30))


def kelly_qty_covered_call(spot: float, strike: float, dte: int, iv: float,
                          uncovered_shares: int, premium: float,
                          max_frac: float = 0.5,
                          kelly_safety: float = 0.5,
                          model_conviction: float = 0.0) -> int:
    """Kelly-sized qty for a covered call.

    For CC: collateral is shares (not cash). Constraint is uncovered_shares
    available. Kelly tells us what fraction of those shares to write CCs on.

    Args:
        uncovered_shares: shares above core_floor not yet covered
    """
    if dte <= 0 or iv <= 0 or strike <= 0 or premium <= 0 or uncovered_shares < 100:
        return 0

    T = dte / 365.0
    d2 = (math.log(spot / strike) - 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
    p_otm_bs = float(norm.cdf(d2))   # prob spot < strike per BS

    # CONVICTION ADJUSTMENT: for CCs, bullish model conviction means we
    # think shares will go UP → call is more likely to be ITM → p_otm DOWN.
    # So we subtract model_conviction (opposite sign vs puts).
    p_otm = max(0.01, min(0.99, p_otm_bs - model_conviction))

    if p_otm < 0.5:
        # Allow if explicitly aggressive ITM CC (force-assignment trade)
        if premium < 0.05:
            return 0
        conviction = 0.5  # half capacity for ITM CCs (force assignment)
    else:
        conviction = (p_otm - 0.5) * 2.0
    f = max(0.0, min(conviction * kelly_safety, max_frac))
    max_shares = uncovered_shares * f
    qty = int(max_shares // 100)
    return max(0, min(qty, uncovered_shares // 100))


if __name__ == '__main__':
    # Sanity tests
    print("=== Kelly sizing sanity tests ===")
    print()
    print("Short put scenarios (UNG @ $11.36, cash $50K):")
    for K, dte, iv in [(10.50, 30, 0.45), (10.50, 45, 0.45), (10.00, 45, 0.45),
                       (11.00, 30, 0.45), (11.00, 30, 0.80), (10.50, 30, 0.80)]:
        # Approximate premium via simple intrinsic + extrinsic
        import math
        T = dte / 365
        d2 = (math.log(11.36 / K) - 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
        p_otm = float(norm.cdf(d2))
        # Rough BS put premium
        from scipy.stats import norm as _n
        d1 = d2 + iv * math.sqrt(T)
        prem = K * math.exp(-0.045 * T) * (1 - float(_n.cdf(d2))) - 11.36 * (1 - float(_n.cdf(d1)))
        prem = max(0.01, prem)
        q = kelly_qty_short_put(11.36, K, dte, iv, 50000, prem)
        print(f"  K=${K:.2f} dte={dte:2d} iv={iv:.2f}  prem=${prem:.3f}  p_otm={p_otm:.2%}  → qty={q}")
    print()
    print("Covered call (1000 uncovered shares):")
    for K, dte, iv in [(11.50, 30, 0.45), (12.00, 30, 0.45), (11.50, 45, 0.45),
                       (12.00, 30, 0.80)]:
        T = dte / 365
        d2 = (math.log(11.36 / K) - 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
        d1 = d2 + iv * math.sqrt(T)
        prem = 11.36 * float(norm.cdf(d1)) - K * math.exp(-0.045 * T) * float(norm.cdf(d2))
        prem = max(0.01, prem)
        q = kelly_qty_covered_call(11.36, K, dte, iv, 1000, prem)
        p_itm = 1 - float(norm.cdf(d2))
        print(f"  K=${K:.2f} dte={dte:2d} iv={iv:.2f}  prem=${prem:.3f}  p_itm={p_itm:.2%}  → qty={q}")
