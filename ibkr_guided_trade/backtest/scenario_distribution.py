"""ScenarioDistribution — production-port (item #2).

7-point quantile kernel covering ±2σ for forward UNG spot distribution.
Simplified vs production: omits outlook anchor + stress tails (would need
more wiring to fundamentals model). Captures the core kernel mechanics.

Usage:
  sd = ScenarioDistribution(spot=11.36, sigma_annual=0.45, z_score=+0.5,
                            contango_per_day=-0.001)
  p = sd.prob_below(strike=10.5, dte=30)   # P(spot < strike at dte)
  e = sd.expected_intrinsic(strike=10.5, dte=30, right='P')

Used by Kelly to replace BS d2 approximation with multi-point quantile.
"""
import math


# 7-point quantile kernel covering ±2σ + 0 (matches production)
_QUANTILES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_Q_WEIGHTS = (0.05, 0.20, 0.20, 0.10, 0.20, 0.20, 0.05)


class ScenarioDistribution:
    """Discrete kernel of forward spot at a horizon. Backtest port of
    production's ScenarioDistribution."""

    def __init__(self, spot: float, sigma_annual: float = 0.45,
                 z_score: float = 0.0,
                 contango_per_day: float = -0.001,
                 seasonal_drift_per_day: float = 0.0,
                 seasonal_vol_scale: float = 1.0):
        self.spot = float(spot)
        self.sigma_annual = float(sigma_annual)
        self.z_score = float(z_score)
        self.contango_per_day = float(contango_per_day)
        self.seasonal_drift_per_day = float(seasonal_drift_per_day)
        self.seasonal_vol_scale = float(seasonal_vol_scale)
        self._cache = {}

    def _build(self, days: int):
        """Return list of (spot, weight) tuples at the given horizon."""
        if days <= 0:
            return [(self.spot, 1.0)]
        if days in self._cache:
            return self._cache[days]
        T = days / 365.0
        sigma_h = self.sigma_annual * math.sqrt(T) * self.seasonal_vol_scale
        # Cyclical-first drift (per production):
        regime_drift = self.seasonal_drift_per_day + self.z_score * 0.0004
        log_spot = math.log(max(0.01, self.spot))
        mu = log_spot + (regime_drift + self.contango_per_day) * days - 0.5 * sigma_h ** 2
        points = [(math.exp(mu + q * sigma_h), w)
                  for q, w in zip(_QUANTILES, _Q_WEIGHTS)]
        # Renorm safety
        total_w = sum(w for _, w in points)
        if total_w > 0:
            points = [(sp, w / total_w) for sp, w in points]
        self._cache[days] = points
        return points

    def prob_below(self, strike: float, dte: int) -> float:
        """P(spot_T < strike) — for put OTM probability inverse."""
        points = self._build(dte)
        return sum(w for sp, w in points if sp < strike)

    def prob_above(self, strike: float, dte: int) -> float:
        points = self._build(dte)
        return sum(w for sp, w in points if sp > strike)

    def expected_intrinsic(self, strike: float, dte: int, right: str) -> float:
        """E[(K-S)+ for puts, (S-K)+ for calls]."""
        points = self._build(dte)
        if right == 'P':
            return sum(w * max(0.0, strike - sp) for sp, w in points)
        else:
            return sum(w * max(0.0, sp - strike) for sp, w in points)


if __name__ == '__main__':
    # Sanity test: compare to BS d2 approximation
    from scipy.stats import norm
    spot, K, dte, iv = 11.36, 10.50, 30, 0.45
    sd = ScenarioDistribution(spot=spot, sigma_annual=iv, z_score=0)
    p_below_sd = sd.prob_below(K, dte)
    e_intrinsic_sd = sd.expected_intrinsic(K, dte, 'P')
    # BS d2
    T = dte / 365
    d2 = (math.log(spot / K) - 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
    p_below_bs = float(norm.cdf(-d2))
    print(f'spot={spot}, K={K}, dte={dte}, iv={iv}, z=0')
    print(f'  P(spot<K) — SD: {p_below_sd:.4f}  BS d2: {p_below_bs:.4f}')
    print(f'  E[(K-S)+] SD: ${e_intrinsic_sd:.4f}')
    print()
    # With bullish z
    sd2 = ScenarioDistribution(spot=spot, sigma_annual=iv, z_score=+1.0)
    p_b2 = sd2.prob_below(K, dte)
    print(f'  With z=+1.0 (bullish): P(spot<K)={p_b2:.4f} (lower = more bullish)')
    sd3 = ScenarioDistribution(spot=spot, sigma_annual=iv, z_score=-1.0)
    p_b3 = sd3.prob_below(K, dte)
    print(f'  With z=-1.0 (bearish): P(spot<K)={p_b3:.4f} (higher = more bearish)')
