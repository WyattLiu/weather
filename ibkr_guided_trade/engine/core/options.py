"""Black-Scholes option pricing — pure functions, used everywhere."""
import math
from scipy.stats import norm


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    if T <= 0.001 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    if T <= 0.001 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.045) -> float:
    if T <= 0.001 or sigma <= 0:
        if right == 'C':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if right == 'C' else norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.045) -> float:
    if T <= 0.001 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(S: float, K: float, T: float, sigma: float, right: str, r: float = 0.045) -> float:
    if T <= 0.001 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -S * sigma * norm.pdf(d1) / (2 * math.sqrt(T))
    if right == 'C':
        return (term1 - r * K * math.exp(-r * T) * norm.cdf(d2)) / 252
    return (term1 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 252
