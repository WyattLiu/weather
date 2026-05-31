"""All tunable engine parameters in ONE place.

Used by both live trading and backtesting. Parameter optimization searches
over instances of this class to find the best configuration.

Defaults match the live ung_visualizer.py settings as of cycle 200+.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Params:
    # ───────── Regime classification (z-score thresholds) ─────────
    z_extreme_cheap: float = 1.0
    z_cheap: float = 0.5
    z_neutral_upper: float = 0.5  # mirror — used for symmetry
    z_neutral_lower: float = -0.5
    z_rich: float = -0.5
    z_extreme_rich: float = -1.0

    # ───────── Z-score factor weights ─────────
    w_storage_level: float = 0.30
    w_days_supply: float = 0.25
    w_ng_trend: float = 0.20
    w_vix: float = 0.10
    w_oil_ng_ratio: float = 0.15

    # ───────── Strategy aggressiveness per regime ─────────
    # OTM % for put writes (0 = ATM, 0.05 = 5% OTM, etc.)
    otm_put_extreme_cheap: float = 0.00   # aggressive ATM
    otm_put_cheap: float = 0.05
    otm_put_neutral: float = 0.10
    otm_put_rich: float = 0.20            # very OTM, defensive
    otm_put_extreme_rich: float = 0.99    # effectively don't sell new puts

    # OTM % for covered calls
    otm_call_extreme_cheap: float = 0.10  # don't cap upside in cheap regime
    otm_call_cheap: float = 0.05
    otm_call_neutral: float = 0.05
    otm_call_rich: float = -0.02          # slightly ITM, force assignment
    otm_call_extreme_rich: float = -0.05  # deeper ITM

    # Number of contracts per write
    put_qty_extreme_cheap: int = 5
    put_qty_cheap: int = 5
    put_qty_neutral: int = 4
    put_qty_rich: int = 1
    put_qty_extreme_rich: int = 0
    call_qty: int = 5

    # ───────── Risk management ─────────
    take_profit_pct: float = 0.50         # close shorts at 50% premium decay
    roll_down_threshold_pct: float = 0.02  # roll when spot 2% below strike
    min_dte_to_roll: int = 5

    # ───────── Forward projection / quality components ─────────
    forward_horizon_days: int = 42
    smoothness_window_weeks: int = 6
    gamma_load_excess_pct: float = 0.20   # only penalize excess vol, not all
    mean_reversion_uplift: float = 0.20   # extra penalty when |z| > 0.5

    # ───────── Cash management ─────────
    boxx_yield: float = 0.04
    boxx_deploy_threshold: float = 5000   # deploy if idle > $5k
    boxx_deploy_fraction: float = 0.70    # 70% of excess
    min_cash_buffer: float = 1000
    never_negative_cash: bool = True

    # ───────── Tactical bearish (EXTREME_RICH only) ─────────
    bearish_stack_enabled: bool = True
    long_put_dte: int = 90
    long_put_qty: int = 3
    long_put_otm_pct: float = 0.05
    kold_nav_fraction: float = 0.03

    # ───────── Friction (WS specifics) ─────────
    commission_per_contract: float = 0.0  # WS = zero
    option_spread_per_share: float = 0.03
    share_spread_per_share: float = 0.005

    # ───────── Beam search ─────────
    beam_width: int = 3
    max_recs: int = 10
    min_marginal_score: float = 3.0
    income_bypass_threshold: float = -50.0
    income_mode_target_pct: float = 0.60  # below 60% target → aggressive

    # ───────── Income target ─────────
    target_weekly_income: float = 1500.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Params':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Default singleton — what the live engine uses
DEFAULT = Params()
