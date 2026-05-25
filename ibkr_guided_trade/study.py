#!/usr/bin/env python3
"""
Options Trading Knowledge Base — Study Script

Everything learned from building and running the S&P 500 bull put spread scanner,
analyzing dozens of stocks, and placing spreads on IBKR and WealthSimple.

Usage:
    python study.py                # Full study guide (all topics)
    python study.py <topic>        # Single topic
    python study.py --list         # List available topics
"""

import sys
import textwrap

# ANSI colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title):
    bar = "═" * 60
    print(f"\n{CYAN}{BOLD}{bar}")
    print(f"  {title}")
    print(f"{bar}{RESET}\n")


def sub(text):
    print(f"  {YELLOW}{text}{RESET}")


def body(text):
    print(textwrap.dedent(text).strip() + "\n")


def example(text):
    print(f"  {GREEN}{text}{RESET}")


def warn(text):
    print(f"  {RED}⚠ {text}{RESET}")


# ─────────────────────────────────────────────────────────────
# Topics
# ─────────────────────────────────────────────────────────────

def topic_spread_mechanics():
    header("SPREAD MECHANICS — Bull Put Spread Anatomy")
    body("""
    A bull put spread is a defined-risk credit spread. You're betting the stock
    stays above your short strike by expiration.

    Construction:
      • SELL a higher-strike put  (collect premium)
      • BUY a lower-strike put   (cap your downside)
      • Same expiry, same underlying
    """)

    sub("P&L Formulas")
    body("""
      Max profit  = net credit received
      Max loss    = spread width − credit
      Break-even  = short strike − credit
    """)

    sub("Example: ORCL 140/135P for $1.15 credit")
    example("  Sell 140P, Buy 135P  →  $5 wide spread")
    example("  Max profit = $1.15  ($115 per contract)")
    example("  Max loss   = $5.00 − $1.15 = $3.85  ($385 per contract)")
    example("  Break-even = $140 − $1.15 = $138.85")
    print()

    body("""
    Both legs expire worthless if the stock stays above the short strike.
    The long put is pure insurance — you never want it to be in-the-money,
    but it caps your loss if the stock craters.

    Time decay (theta) works in your favor. Every day that passes with the
    stock above your short strike, both puts lose value and your spread
    becomes cheaper to close.
    """)


def topic_iv_rv_edge():
    header("IV vs RV EDGE — The Core Concept")
    body("""
    The fundamental question for option selling: are options overpriced
    relative to how much the stock actually moves?

      IV  = Implied Volatility  (what the market prices in)
      RV  = Realized Volatility (what actually happened)
    """)

    sub("Positive Edge = Options Overpriced = Selling Has +EV")
    body("""
      edge = ATM_IV − RV30

    If edge is positive, the market is pricing in more movement than the
    stock has actually delivered over the last 30 trading days.
    """)

    sub("Computing RV from Daily Closes")
    body("""
      1. Get 1 year of daily closing prices
      2. Compute log returns: r_i = ln(close_i / close_{i-1})
      3. Take stdev of last N returns (e.g., N=30 for RV30)
      4. Annualize: RV = stdev × √252

    The √252 factor converts daily vol to annual vol (252 trading days/year).
    """)

    sub("What Makes a Good Edge?")
    example("  Edge > 15%  →  Excellent (e.g., DASH +34.7%, DG +17.9%)")
    example("  Edge 10-15% →  Good (e.g., ORCL +12.3%, TGT +13.8%)")
    example("  Edge 5-10%  →  Moderate (worth scanning if liquidity is good)")
    example("  Edge < 5%   →  Not enough to compensate for tail risk")
    print()

    body("""
    Edge alone isn't enough — you also need liquidity (OI), sufficient
    OTM distance, and manageable width. A stock with 30% edge but OI of
    50 is untradeable.
    """)


def topic_scanning():
    header("SCANNING — 3-Phase Methodology")

    sub("Phase 1: Historical RV Screen (fast, ~5 min for 503 stocks)")
    body("""
    Request 1-year daily bars for all S&P 500 stocks via IBKR.
    Compute RV10, RV30, RV60 from log returns.
    Filter to stocks with RV30 > some threshold.
    Also compute recent drop (20-day return) to find post-crash candidates.

    Batch 50 requests at a time, 2s delay between batches.
    IBKR rate limits: ~50 simultaneous historical data requests.
    """)

    sub("Phase 2: ATM IV Screen (~10 min for ~100 candidates)")
    body("""
    For each Phase 1 candidate, request the ATM put option at target DTE.
    Compute IV-RV edge.
    Sort by edge descending.
    Top 50 advance to Phase 3 (or deep-scan individually).

    Use nearest expiry to target DTE (e.g., 35 DTE → Mar 20).
    ATM strike = nearest to current spot price.
    """)

    sub("Phase 3: Full Put Chain Analysis (2-3 min per stock)")
    body("""
    For top picks, pull the entire put chain at target expiry.
    For each strike, collect: bid, ask, OI, last price, IV.
    Compute for each potential spread:
      • Credit (mid and natural)
      • Width, max loss, RoR
      • OTM distance
      • OI grade on both legs

    IBKR quirk: some stocks have half-dollar strikes ($142.50).
    IBKR quirk: OI requires generic tick types 101 and 106 — use
    streaming (not snapshot) to get reliable OI data.
    """)

    sub("Scanner Script")
    example("  python scan_spreads.py                  # Full 3-phase scan")
    example("  python scan_spreads.py --ticker GOOGL   # Deep scan one stock")
    example("  python scan_spreads.py --top 20         # Show top 20 by edge")
    example("  python scan_spreads.py --quick          # Curated high-IV list only")
    example("  python scan_spreads.py --save           # Save results to CSV")
    print()


def topic_liquidity():
    header("LIQUIDITY — Open Interest Grading")
    body("""
    Open Interest (OI) is the number of outstanding contracts at a given
    strike. It's the single best proxy for how easily you can get filled.
    """)

    sub("OI Grading Scale")
    example("  Grade A:  OI > 3,000   →  Tight spreads, instant fills at mid")
    example("  Grade B:  OI > 500     →  Decent. May need patience for mid fill")
    example("  Grade C:  OI 100-500   →  Thin. Expect slippage, use limit orders")
    example("  Avoid:    OI < 100     →  Untradeable. Don't bother.")
    print()

    sub("Real Examples from Feb 2026 Scan")
    body("""
    ORCL 140P:  OI 17,818  →  Grade A++, instant fill
    AVGO 290P:  OI 5,611   →  Grade A, tight spread
    WBD 27P:    OI 160,958 →  Best in universe
    DASH 140P:  OI 254     →  Thin but fillable with patience
    AXON 245P:  OI 79      →  Untradeable despite 16.7% edge
    ULTA 310P:  OI 17      →  Completely untradeable
    """)

    sub("Spread Width and Natural vs Mid")
    body("""
    Bid-ask spread on options is often $0.10-$0.50 wide.

    Natural = price you get instantly (sell at bid, buy at ask).
    Mid     = halfway between bid and ask. Better price, needs patience.

    For Grade A OI: mid fill usually happens in seconds.
    For Grade B OI: mid fill may take minutes to hours.
    For Grade C OI: you may need to walk from mid toward natural.

    On a $1.15 credit spread, the difference between mid ($1.15) and
    natural ($0.90) is $25/contract — significant on small spreads.
    """)


def topic_selection():
    header("SELECTION — Criteria for Actionable Spreads")

    sub("Minimum Thresholds")
    body("""
      Edge (IV − RV):    > 10%
      OTM distance:      > 8%  (preferably > 10%)
      OI (both legs):    > 500  (Grade B minimum)
      Return on Risk:    > 25%
      Spread width:      $5 − $10  (sweet spot)
    """)

    sub("Scoring Components")
    body("""
    When multiple spreads pass thresholds, rank by composite score:

      1. OTM distance  — farther = safer, but less credit
      2. RoR           — higher = better payoff per dollar risked
      3. OI grade      — affects fill quality and exit ability
      4. IV-RV edge    — core statistical advantage

    There's always a trade-off: moving further OTM reduces credit
    and RoR but increases probability of profit.
    """)

    sub("From the Feb 2026 Scan")
    body("""
    503 stocks scanned → 50 had positive edge → 8 deep-scanned →
    4 made the final portfolio.

    Many stocks with good edge were eliminated by OI:
      AXON: 16.7% edge, OI 79/70 → untradeable
      PODD: 17.5% edge, OI 27/21 → untradeable
      ULTA: 14.7% edge, OI 17/67 → untradeable

    Some eliminated by OTM distance:
      WBD:  18.1% edge, only 3.9% OTM → too close
      NFLX: 11.2% edge, only 3.1% OTM → too close (post-split)

    The final 4 (ORCL, AVGO, DASH, TGT) balanced all criteria.
    """)


def topic_risk_mgmt():
    header("RISK MANAGEMENT — Rules for Open Positions")

    sub("Exit Rules")
    body("""
      1. Close at 50% profit if reached early in the trade.
         e.g., sold for $1.15 → close when you can buy back for $0.57

      2. Close at 14 DTE if not yet at 50% profit.
         Gamma risk accelerates in the final 2 weeks — small moves
         cause large P&L swings. Don't hold through this.

      3. Single expiry simplifies management.
         All positions expire on the same day → one monitoring date.
    """)

    sub("Don't Panic")
    body("""
    If a stock drops 5% in one day:
      • Your IV edge protects you — options were overpriced
      • Check if spot is still above your short strike
      • Check how many DTE remain
      • If > 14 DTE and stock is still OTM, hold

    Max loss requires ALL positions in your portfolio to breach
    their short strikes simultaneously. This is extremely unlikely
    for a diversified set of uncorrelated names.
    """)

    sub("Position Sizing")
    body("""
    Keep max risk per position roughly equal (~$1000 each).
    This means:
      • 2x contracts on $5 wide spreads  (max loss $770-$850)
      • 1x contracts on $10 wide spreads (max loss $670-$770)

    Total portfolio max risk should be an amount you're comfortable
    losing entirely, even though full loss is very unlikely.
    """)


def topic_ws_trading():
    header("WEALTHSIMPLE TRADING — Multi-Leg Patterns")

    sub("Critical: timeInForce Must Be Uppercase")
    warn('timeInForce: "DAY" works.  "day" → UNPROCESSABLE_ENTITY')
    print()

    sub("Credit vs Debit Orders")
    body("""
    In place_multileg_order():
      Credit spread (selling):  limit_price = NEGATIVE  (you receive money)
      Debit spread (buying):    limit_price = POSITIVE  (you pay money)
    """)

    sub("Cancelling Multi-Leg Orders")
    body("""
    Each leg has its own order ID:  {ext_id}-leg-{i}  for i in 1..N
    Cancel each leg individually.
    """)

    sub("Token Refresh")
    body("""
    WS access tokens expire frequently. For long-running loops:
      ws.refresh_access_token(oauth_data, device_id)
      ws.update_cookies_with_new_token()
    Refresh proactively every ~10 minutes.
    """)

    sub("Security Search")
    body("""
    To find security IDs for new symbols:
      Use QUERY_SECURITY_SEARCH with query parameter.
      Returns sec IDs needed for order placement.
    """)

    warn("After-hours multileg orders → INTERNAL_SERVER_ERROR")
    warn("DAY orders expire at market close — no GTC for multileg")
    print()

    sub("WS Commands")
    example("  python ws_trading.py status                        # Account balances")
    example("  python ws_trading.py positions                     # Current positions")
    example("  python ws_trading.py orders                        # Recent orders")
    example("  python ws_trading.py opt-expiry SPY                # Option expiry dates")
    example("  python ws_trading.py opt-chain SPY 2026-02-20 PUT  # Put chain")
    example("  python ws_trading.py straddle SPY 2026-02-20 692 18.50  # Buy straddle")
    example("  python ws_trading.py multileg-status <order-id>    # Check multileg")
    print()


def topic_ibkr_trading():
    header("IBKR TRADING — Spread Placement & Management")

    sub("Client ID Matters")
    body("""
    Each ib_insync connection uses a client ID. If you use random client IDs,
    each reconnection creates a "new" client that can't see previous orders.
    This leads to duplicate orders stacking up.

    Solution: use a fixed client ID (50) for all trading operations.
    """)

    sub("Viewing and Cancelling Orders")
    body("""
    ib.reqAllOpenOrders()  — fetch ALL orders from IBKR server
                             (not just from current client)
    ib.reqGlobalCancel()   — cancel ALL orders regardless of client
    """)

    sub("OI Data Requires Streaming")
    body("""
    Generic tick types 101 (shortable) and 106 (OI) must be requested.
    Use streaming mode, NOT snapshot. Snapshots don't reliably return OI.
    """)

    sub("Spread Command — Three Modes")
    body("""
    The `spread` command has three distinct modes:

      1. CREDIT (default): Sell short_strike, buy long_strike → receive credit
         Used for: bull put spreads, bear call spreads

      2. DEBIT (--open-debit): Buy short_strike, sell long_strike → pay debit
         Used for: bear put spreads, bull call spreads, long IC legs

      3. CLOSE (--close): Auto-detects position direction from IBKR positions
         and reverses it correctly. No need to remember which way you opened.
    """)

    warn("LESSON: Never use --close to OPEN a debit spread!")
    body("""
    Before the fix, --close was a simple flag that reversed leg direction.
    It was misused to open debit spreads (since no debit mode existed).
    But then closing those positions with --close reversed AGAIN, doubling
    the position instead of closing it.

    The NVDA hybrid trade accidentally doubled put spreads this way.
    (Lucky break: NVDA dropped and the extra puts made +$185 profit.)

    The fix: --close now reads actual IBKR positions to determine which
    direction you hold, then always reverses correctly. --open-debit is
    the proper way to open debit spreads.
    """)

    sub("IBKR Commands")
    example("  python ibkr_trading.py status                      # Open orders + positions")
    example("  python ibkr_trading.py snapshot                    # Full account P&L")
    example("  python ibkr_trading.py cancel-all                  # Cancel all orders")
    example("  python ibkr_trading.py quote ORCL                  # Stock quote")
    example("  python ibkr_trading.py opt-chain ORCL 35           # Options chain ~35 DTE")
    example("  python ibkr_trading.py scan-puts ORCL              # Scan 10-45 DTE puts")
    example("  python ibkr_trading.py spread ORCL 20260320 P 140 135              # Credit spread")
    example("  python ibkr_trading.py spread NVDA 20260227 P 190 185 --open-debit # Debit spread")
    example("  python ibkr_trading.py spread ORCL 20260320 P 140 135 --close      # Close (auto)")
    print()


def topic_scan_results():
    header("SCAN RESULTS — Feb 2026 S&P 500 Findings")

    sub("Universe: 503 S&P 500 stocks scanned via IBKR")
    body("""
    Phase 1 → Phase 2: 50 stocks had positive IV-RV edge
    Phase 2 → Phase 3: Top 8 deep-scanned for full spread analysis
    Phase 3 → Portfolio: 4 trades selected
    """)

    sub("Tier 1 — High Edge + Good Liquidity (Final Portfolio)")
    body("""
    ┌──────┬──────────────┬────────┬────────┬───────┬───────┬──────────┐
    │  #   │ Trade        │ Credit │ RoR    │ Edge  │ OTM   │ OI Grade │
    ├──────┼──────────────┼────────┼────────┼───────┼───────┼──────────┤
    │  1   │ DASH 140/135P│ $1.58  │ 46.0%  │+34.7% │ 12.7% │ C (254)  │
    │  2   │ AVGO 290/280P│ $2.33  │ 30.4%  │+17.7% │ 11.3% │ A (5611) │
    │  3   │ ORCL 140/135P│ $1.15  │ 29.9%  │+12.3% │ 13.1% │ A+(17818)│
    │  4   │ TGT 105/100P │ $1.05  │ 26.5%  │+13.8% │  9.5% │ B (2031) │
    └──────┴──────────────┴────────┴────────┴───────┴───────┴──────────┘
    """)

    sub("Notable Rejections")
    body("""
    High edge, no liquidity:
      AXON +16.7% edge, OI 79/70     → untradeable
      PODD +17.5% edge, OI 27/21     → untradeable
      WSM  +14.1% edge, OI 37/118    → untradeable
      ULTA +14.7% edge, OI 17/67     → untradeable
      KEYS +12.9% edge, OI 12/43     → untradeable

    Good edge, too close to the money:
      WBD  +18.1% edge, only 3.9% OTM (but incredible 161k OI)
      NFLX +11.2% edge, only 3.1% OTM (post-split tiny spreads)
      DG   +17.9% edge, only 4.0% OTM

    Phase 2 only (not deep-scanned): GPN, DPZ, LYV, CRL, CRM, TAP...
    See bull_put_spread_plan.md for the complete write-up.
    """)


def topic_forward_atm():
    header("FORWARD ATM — Put-Call Parity & Straddle Pricing")

    sub("Forward Price via Put-Call Parity")
    body("""
    The forward price F at a given strike K is:

      F = K + C - P

    where C = call mid, P = put mid at strike K.
    """)
    warn("It is NOT (C+P)/2 + K. That formula is wrong.")
    print()

    sub("Finding the ATM Strike")
    body("""
    The true ATM strike is where |call_mid − put_mid| is smallest.
    At this strike, calls and puts are closest in value.

    Once you have the ATM strike, compute F = K_atm + C_atm − P_atm.
    """)

    sub("Brenner-Subrahmanyam Straddle Approximation")
    body("""
    Quick estimate of ATM straddle value:

      Straddle ≈ 0.798 × F × σ × √T

    where:
      F = forward price
      σ = implied volatility (annualized, decimal)
      T = time to expiry in years (DTE / 365)

    This is useful for sanity-checking whether straddle prices make
    sense relative to IV, or for estimating fair value when bid-ask
    is wide.
    """)

    sub("Example")
    example("  SPY at $592, 28 DTE, IV = 18%")
    example("  T = 28/365 = 0.0767")
    example("  Straddle ≈ 0.798 × 592 × 0.18 × √0.0767")
    example("         ≈ 0.798 × 592 × 0.18 × 0.277")
    example("         ≈ $23.56")
    print()


def topic_vol_concepts():
    header("VOLATILITY CONCEPTS — RV Windows, IV Crush, Term Structure")

    sub("Realized Volatility Windows")
    body("""
    RV is typically measured over different lookback periods:

      RV10  = last 10 trading days  (~2 weeks)  — short-term, noisy
      RV30  = last 30 trading days  (~6 weeks)  — standard reference
      RV60  = last 60 trading days  (~3 months) — medium-term trend

    Declining RV (RV10 < RV30 < RV60) is bullish for selling options:
    it means the stock is calming down, but IV hasn't caught up yet.

    Example from scan: AVGO had RV60=51% → RV30=42% (declining).
    """)

    sub("Post-Crash Elevated IV")
    body("""
    After a large drop, IV spikes because:
      1. Realized vol just spiked (recent big moves)
      2. Fear/demand for puts increases
      3. Market-makers widen spreads

    This is the CLASSIC option-selling opportunity:
      • The crash already happened (realized vol is high)
      • IV is elevated, pricing in continued chaos
      • But historically, vol tends to mean-revert

    Stocks like ORCL (-33% in 20d) and CRL (-28.8% in 20d) had
    elevated IV post-crash — exactly when selling edge is largest.
    """)

    sub("IV Crush After Earnings")
    body("""
    IV spikes before earnings announcements as uncertainty is highest.
    After earnings, uncertainty resolves and IV drops sharply ("crush").

    If you sell options before earnings, IV crush benefits you BUT
    the actual move might exceed what IV priced in.

    For spread scanning, we generally AVOID stocks with earnings
    within our DTE window unless we specifically want that exposure.
    """)


def topic_reverse_calendar():
    header("REVERSE CALENDAR — Buy Near-Dated, Sell Far-Dated")

    sub("What Is a Reverse Calendar?")
    body("""
    A reverse calendar (RC) is a volatility play with 4 legs:

      BUY  near-dated put   (e.g., 0DTE or 2DTE)
      BUY  near-dated call  (same expiry as above)
      SELL far-dated put    (e.g., 7DTE or 9DTE)
      SELL far-dated call   (same expiry as above)

    You're buying a straddle on the near expiry and selling a straddle
    on the far expiry. If entered for a credit, your max loss is when
    the stock doesn't move (both straddles collapse, but far-dated
    retains more value since it has more time).
    """)

    sub("Why It Works for Earnings")
    body("""
    Before earnings, near-dated options have MUCH higher IV than
    far-dated options (the "earnings premium"):

      Pre-earnings:
        0DTE IV: ~210%    (all about the binary event)
        7DTE IV: ~95%     (event is a fraction of its remaining life)

      Post-earnings:
        0DTE: expires at intrinsic (IV irrelevant)
        7DTE: IV crushes to ~60%   (uncertainty resolved)

    The RC profits because:
      1. Near-dated captures the MOVE (big delta/gamma from straddle)
      2. Far-dated profits from IV CRUSH (you sold at 95%, it drops to 60%)
      3. Net effect: you profit from large moves in either direction
    """)

    sub("0DTE vs 2DTE Near Leg")
    body("""
    0DTE near leg (ideal):
      • Expires at intrinsic — no extrinsic to worry about
      • Clean P&L: near straddle value = |spot − strike| at close
      • Maximum IV differential vs far leg
      • Used in INTC trade (0DTE/7DTE)

    2DTE near leg (less ideal):
      • Still has extrinsic value after earnings
      • Near straddle doesn't fully "resolve" at intrinsic
      • IV crush hits your long options too (partially offsets)
      • Wider break-even needed for same profit
      • Used in NVDA trade (2DTE/9DTE — no 0DTE available)
    """)

    sub("Case Study: INTC RC (Jan 22-23, 2026)")
    example("  Position:  2x ($52P/$55C) 0DTE/7DTE")
    example("  Entry:     $1.34 avg credit per combo ($267 total)")
    example("  IV gap:    0DTE ~210% vs 7DTE ~95%")
    example("  INTC move: $54 → $47 (-13%)")
    example("  Exit:      $0.69 debit per combo")
    example("  Result:    +$154.68 profit (3.9% on margin)")
    print()

    sub("P&L Shape")
    body("""
    The RC has a "valley" P&L shape — you lose money in the middle
    and profit on the wings:

      Big move down:  near puts go deep ITM → profit
      Big move up:    near calls go deep ITM → profit
      Flat:           far straddle retains more value than near → loss

    Break-even is typically at ±5-7% move (depends on IV differential
    and credit received). Max loss occurs at exactly flat.
    """)

    sub("IBKR RC Command")
    example("  # Open (for credit):")
    example("  python ibkr_trading.py rc INTC 20260123 20260130 52 55 --credit 1.30")
    example("")
    example("  # Close (for debit):")
    example("  python ibkr_trading.py rc INTC 20260123 20260130 52 55 --close --debit 0.55")
    example("")
    example("  # Preview without placing:")
    example("  python ibkr_trading.py rc NVDA 20260227 20260306 190 205 --dry-run")
    print()

    warn("RC requires significant margin for naked short straddle on far leg")
    warn("IBKR margin ~1.7x more conservative than theoretical Reg-T")
    print()


def topic_iron_condors():
    header("IRON CONDORS — Defined-Risk Multi-Leg Strategies")

    sub("Short Iron Condor (Credit, Profits From Flat)")
    body("""
    Construction:
      SELL OTM put   (e.g., 190P)
      BUY  lower put (e.g., 185P)   ← defines max loss on downside
      SELL OTM call  (e.g., 205C)
      BUY  higher call (e.g., 210C) ← defines max loss on upside

    Essentially: bull put spread + bear call spread at the same time.

    Max profit = total credit (both spreads)
    Max loss   = wider spread width − total credit
    Profits when stock stays between the short strikes.
    Time decay (theta) works for you.
    """)

    sub("Long (Reverse) Iron Condor (Debit, Profits From Big Move)")
    body("""
    Flip every leg of the short IC:
      BUY  OTM put    (e.g., 190P)
      SELL lower put   (e.g., 185P)
      BUY  OTM call   (e.g., 205C)
      SELL higher call (e.g., 210C)

    Essentially: bear put spread + bull call spread.

    Max loss   = total debit paid
    Max profit = spread width − debit
    Profits when stock makes a BIG move past either short strike.
    Time decay (theta) works AGAINST you.
    """)

    sub("When to Use Each")
    body("""
    Short IC (sell premium):
      • High IV environment you expect to normalize
      • Range-bound stock
      • After earnings (IV just crushed, selling what's left)

    Long IC (buy premium):
      • Expecting a large move (earnings, FDA, macro event)
      • Low IV you expect to spike
      • Binary event catalyst
    """)
    warn("Long IC is hurt by IV crush — if you buy before earnings")
    warn("and IV drops, your options lose value even if the move is moderate")
    print()

    sub("Key Metrics")
    body("""
      Credit/Debit: net premium received or paid
      Width:        strike distance on each spread ($5, $10, etc.)
      Max risk:     width − credit (short IC) or debit paid (long IC)
      Break-even:   short strike ± credit received/paid
      Margin:       max loss per spread × 100 (defined risk = max loss)
    """)


def topic_hybrid_strategy():
    header("HYBRID STRATEGY — Defined-Risk Reverse Calendar")

    sub("The Problem")
    body("""
    Reverse calendars are the best earnings play (profit from move + IV crush)
    but they require naked short straddles on the far leg → high margin.

    NVDA example: RC margin was $6,012 but account only had $5,274.

    Iron condors have defined risk (low margin) but:
      • Short IC profits only from flat (wrong for earnings)
      • Long IC profits from moves BUT IV crush hurts your long options
    """)

    sub("The Solution: Hybrid = Long IC (Near) + Short IC (Far)")
    body("""
    Combine TWO iron condors across different expirations:

      Near-dated IC (long/debit): captures the MOVE via gamma
        BUY  Feb 27 190P / SELL Feb 27 185P  (bear put debit spread)
        BUY  Feb 27 205C / SELL Feb 27 210C  (bull call debit spread)

      Far-dated IC (short/credit): captures IV CRUSH
        SELL Mar 6 190P / BUY Mar 6 185P     (bull put credit spread)
        SELL Mar 6 205C / BUY Mar 6 210C     (bear call credit spread)

    This is a DEFINED-RISK version of the reverse calendar.
    Same P&L shape (valley), but max loss is capped by the spread widths.
    """)

    sub("Why Each Leg Works")
    body("""
    Near IC (long): You own the straddle-like structure.
      • If NVDA moves ±8%, your put or call spread goes near max value
      • Yes, IV crush hurts these — but delta/gamma profits OUTWEIGH
        the IV loss on a big enough move
      • 2DTE means very high gamma — options react strongly to movement

    Far IC (short): You sold the straddle-like structure.
      • You sold at elevated pre-earnings IV (~58% for Mar 6)
      • After earnings, IV crushes to ~43%
      • The options you sold drop in value → profit
      • Even if the stock moves, the far-dated spread changes slowly
        (lower gamma than near-dated)
    """)

    sub("Case Study: NVDA Hybrid A (Feb 25-26, 2026)")
    example("  Near IC (Feb 27, 2DTE) — DEBIT legs:")
    example("    +1 Feb 27 190P @ $2.99 / -1 185P @ $1.70  (debit $1.29)")
    example("    +1 Feb 27 205C @ $2.42 / -1 210C @ $1.25  (debit $1.17)")
    example("")
    example("  Far IC (Mar 6, 9DTE) — CREDIT legs:")
    example("    -1 Mar 6 190P @ $4.44 / +1 185P @ $3.06   (credit $1.38)")
    example("    -1 Mar 6 205C @ $3.64 / +1 210C @ $2.23   (credit $1.41)")
    example("")
    example("  Net: $0.33 credit ($33 received)")
    example("  Max loss: -$171 at flat | Max gain: +$147 at ±9%")
    example("  Break-even: ~±5% move ($188 / $207)")
    example("  Margin: $500 (defined risk)")
    print()

    sub("Hybrid vs Pure RC Comparison")
    body("""
    ┌─────────────┬──────────────────┬──────────────────┐
    │             │ Pure RC          │ Hybrid           │
    ├─────────────┼──────────────────┼──────────────────┤
    │ Margin      │ ~$6,000 (naked)  │ ~$500 (defined)  │
    │ Max loss    │ Unlimited*       │ $171 (capped)    │
    │ Max gain    │ ~$265            │ $147             │
    │ Break-even  │ ±5.8%            │ ±5.0%            │
    │ Legs        │ 4                │ 8                │
    │ Complexity  │ Medium           │ High             │
    └─────────────┴──────────────────┴──────────────────┘
    * RC max loss is bounded by the straddle width, not truly unlimited

    Trade-off: hybrid costs less margin and has defined risk, but
    slightly worse payoff and wider break-even. Worth it when capital
    is limited.
    """)

    sub("Execution Notes")
    body("""
    Place credit legs (far IC) first — you receive cash to fund debits.
    Place debit legs (near IC) second.

    On IBKR, use the spread command:
      Credit: python ibkr_trading.py spread NVDA 20260306 P 190 185
      Debit:  python ibkr_trading.py spread NVDA 20260227 P 190 185 --open-debit
      Close:  python ibkr_trading.py spread NVDA 20260306 P 190 185 --close
              (auto-detects direction from positions)

    Close all 8 legs the morning after earnings once the move + crush
    have played out (typically by 10:00-10:30 AM).
    """)


def topic_earnings_trading():
    header("EARNINGS TRADING — IV Term Structure & Timing")

    sub("Pre-Earnings IV Term Structure")
    body("""
    Before a big earnings report, IV across expirations looks like:

      Expiry     DTE    IV      Why
      ─────────  ─────  ──────  ───────────────────────
      Feb 27     2DTE   ~100%   Entire life is the event
      Mar 6      9DTE   ~58%    Event is ~20% of its life
      Mar 20     23DTE  ~45%    Event is ~9% of its life
      Apr 17     51DTE  ~40%    Event is ~4% of its life

    The "earnings premium" is the gap between near and far IV.
    NVDA Feb 2026: ~41% IV gap between 2DTE and 9DTE.
    INTC Jan 2026: ~115% IV gap between 0DTE (210%) and 7DTE (95%).
    """)

    sub("Post-Earnings IV Crush Model")
    body("""
    After the event, uncertainty resolves and IV collapses:

      Pre     →  Post       Crush
      ──────────────────────────────
      0DTE:  expires at intrinsic (IV irrelevant)
      2DTE:  100% → ~40%    (-60 pts)
      7DTE:   95% → ~60%    (-35 pts)
      9DTE:   58% → ~43%    (-15 pts)
      14DTE:  50% → ~45%    (-5 pts)
      28DTE:  42% → ~40%    (-2 pts)

    Key insight: near-dated options crush MUCH harder than far-dated.
    This is what reverse calendars and hybrids exploit.
    """)

    sub("Market-Implied Earnings Move")
    body("""
    The ATM straddle price at the nearest post-earnings expiry tells
    you what the market expects:

      Expected move ≈ ATM straddle price / spot price

    For NVDA at $197 with $11.40 straddle (Feb 27):
      Implied move ≈ $11.40 / $197 = 5.8%

    But this includes non-earnings vol. To isolate the earnings-only
    component, subtract the "normal" daily vol:

      Straddle implied σ ≈ straddle / (0.798 × F × √T)
      Normal vol component = σ_far × √T
      Earnings component = √(straddle_var − normal_var)

    For NVDA: total 5.8% implied, ~3.5% was earnings-only.
    """)

    sub("Timing")
    body("""
    Entry timing:
      • IDEAL: 15-30 minutes before close on earnings day
      • IV is highest just before close (last chance to trade the event)
      • INTC: entered mid-day (ok but not optimal)
      • NVDA: entered mid-day (market hours, had to work with it)

    Exit timing:
      • Morning after earnings, 9:30-10:30 AM
      • IV crush is usually complete by 10:00 AM
      • Don't rush — let the opening volatility settle (first 5-10 min)
      • INTC: exited 9:35 AM (good timing)

    Day of week matters:
      • Monday earnings → Tuesday exit (clean)
      • Wednesday earnings → 0DTE is same-day (ideal for RC)
      • Friday earnings → weekend decay complicates things
    """)

    sub("Strategy Selection by Account Size")
    body("""
    ┌───────────────────┬─────────────────┬──────────────────┐
    │ Account Size      │ Strategy        │ Margin Needed    │
    ├───────────────────┼─────────────────┼──────────────────┤
    │ < $5,000          │ Hybrid IC       │ ~$500            │
    │ $5,000 - $15,000  │ Reverse Calendar│ ~$5,000-$8,000   │
    │ > $15,000         │ RC + size up    │ Scale with edge  │
    │ Any               │ Long straddle   │ Debit only       │
    └───────────────────┴─────────────────┴──────────────────┘

    The hybrid gives you RC-like exposure at 1/10th the margin.
    Pure RC is cleaner (4 legs vs 8) and has better payoff per dollar.
    """)


def topic_margin():
    header("MARGIN — Defined vs Undefined Risk")

    sub("Defined Risk (Spreads)")
    body("""
    When both legs of a spread are filled, max loss is known:

      Bull put spread: max loss = width − credit
      Bear call spread: max loss = width − credit
      Iron condor: max loss = wider side's width − total credit

    Margin required = max loss × 100 × quantity
    No additional margin surprises.
    """)

    sub("Example")
    example("  ORCL 140/135P for $1.15 credit:")
    example("  Max loss = ($5 − $1.15) × 100 = $385 per contract")
    example("  Margin = $385 per contract")
    example("")
    example("  NVDA hybrid (8 legs, $5 wide each side):")
    example("  Max loss on put side = $500 (debit) − credit from far IC")
    example("  Margin ≈ $500 total (defined risk both sides)")
    print()

    sub("Undefined Risk (Naked Options, RC)")
    body("""
    Naked short options and reverse calendars have theoretically
    large max loss:

      Naked short put: max loss = strike × 100 (stock goes to $0)
      Naked short call: max loss = unlimited (stock goes to ∞)
      RC (naked straddle on far leg): large but bounded by straddle width

    IBKR Reg-T margin formula for naked options:
      max(20% × spot − OTM amount, 10% × strike) + option premium

    For NVDA at $197 with $190P/$205C RC:
      Theoretical Reg-T: ~$4,000-$5,000
      Actual IBKR margin: $6,012 (1.2-1.7x more conservative)
    """)
    warn("IBKR margin is MORE conservative than textbook Reg-T")
    warn("Always check with --dry-run before committing to a trade")
    print()

    sub("Capital Efficiency Comparison")
    body("""
    Same trade, different structures:

    NVDA earnings play targeting ±5% move:

      Long straddle:  $1,140 debit, max loss $1,140, no margin
      Reverse calendar: $0 net (credit), margin ~$6,000
      Hybrid IC:      $33 credit, margin ~$500, max loss $171

    Return on capital deployed:
      Straddle at +8% move: ~$200 / $1,140 = 17.5%
      RC at +8% move:       ~$200 / $6,000 = 3.3%
      Hybrid at +8% move:   ~$133 / $500   = 26.6%

    Hybrid wins on capital efficiency when you have limited funds.
    RC wins on raw P&L when margin is available.
    """)

    sub("Margin Gotchas")
    body("""
    1. IBKR may reject orders even if your math says you have enough
       — they use real-time risk calculations, not textbook formulas

    2. Combo orders (BAG) have different margin treatment than
       individual legs — a 4-leg RC is margined as a unit

    3. If one leg of a spread fails to fill, you're left with a
       naked option and much higher margin — always use combo orders

    4. After-hours margin requirements can be higher than during
       regular hours

    5. Account equity fluctuates with positions — a losing day can
       reduce available margin below what you need for new trades
    """)


def topic_commands():
    header("COMMAND REFERENCE — Quick Reference Card")

    sub("Scanner (scan_spreads.py)")
    example("  python scan_spreads.py                     # Full 3-phase scan")
    example("  python scan_spreads.py --ticker GOOGL      # Deep scan one stock")
    example("  python scan_spreads.py --top 20            # Top 20 by edge")
    example("  python scan_spreads.py --quick             # Curated high-IV list")
    example("  python scan_spreads.py --dte 35            # Target DTE")
    example("  python scan_spreads.py --min-edge 3        # Min edge filter")
    example("  python scan_spreads.py --save              # Save to CSV")
    print()

    sub("IBKR (ibkr_trading.py)")
    example("  python ibkr_trading.py status              # Orders + positions")
    example("  python ibkr_trading.py snapshot            # Full account P&L")
    example("  python ibkr_trading.py cancel-all          # Cancel all orders")
    example("  python ibkr_trading.py quote ORCL          # Stock quote")
    example("  python ibkr_trading.py opt-chain ORCL 35   # Options chain")
    example("  python ibkr_trading.py scan-puts ORCL      # Multi-DTE put scan")
    example("  python ibkr_trading.py scan-calls ORCL     # Multi-DTE call scan")
    example("  python ibkr_trading.py spread ORCL 20260320 P 140 135")
    example("  python ibkr_trading.py spread ORCL 20260320 P 140 135 --close")
    example("  python ibkr_trading.py rc INTC 20260220 20260320 21 24")
    example("  python ibkr_trading.py sell-put UNG 20260116 11 0.35")
    example("  python ibkr_trading.py greeks              # Portfolio Greeks")
    example("  python ibkr_trading.py risk                # Risk analysis")
    print()

    sub("WealthSimple (ws_trading.py)")
    example("  python ws_trading.py status                # Account balances")
    example("  python ws_trading.py positions             # Current positions")
    example("  python ws_trading.py orders                # Recent orders")
    example("  python ws_trading.py opt-expiry SPY        # Option expiry dates")
    example("  python ws_trading.py opt-chain SPY 2026-02-20 PUT")
    example("  python ws_trading.py buy-opt <sec-id> 1 0.50")
    example("  python ws_trading.py sell-opt <sec-id> 1 0.75")
    example("  python ws_trading.py straddle SPY 2026-02-20 692 18.50")
    example("  python ws_trading.py straddle SPY 2026-02-20 692 18.50 --close")
    example("  python ws_trading.py multileg-status <order-id>")
    print()

    sub("Study (this script)")
    example("  python study.py                            # Full study guide")
    example("  python study.py <topic>                    # Single topic")
    example("  python study.py --list                     # List topics")
    print()


# ─────────────────────────────────────────────────────────────
# Topic registry
# ─────────────────────────────────────────────────────────────

TOPICS = [
    ("spread-mechanics", "Bull put spread anatomy & P&L formulas", topic_spread_mechanics),
    ("iv-rv-edge",       "IV vs RV concept — the core selling edge", topic_iv_rv_edge),
    ("scanning",         "3-phase scan methodology", topic_scanning),
    ("liquidity",        "Open Interest grading & fill expectations", topic_liquidity),
    ("selection",        "Criteria for actionable spreads", topic_selection),
    ("risk-mgmt",        "Exit rules & position management", topic_risk_mgmt),
    ("reverse-calendar", "Buy near-dated, sell far-dated for earnings", topic_reverse_calendar),
    ("iron-condors",     "Short IC (credit) & Long IC (debit) structures", topic_iron_condors),
    ("hybrid-strategy",  "Defined-risk RC using two iron condors", topic_hybrid_strategy),
    ("earnings-trading", "IV term structure, timing & strategy selection", topic_earnings_trading),
    ("margin",           "Defined vs undefined risk, capital efficiency", topic_margin),
    ("ws-trading",       "WealthSimple multi-leg patterns & gotchas", topic_ws_trading),
    ("ibkr-trading",     "IBKR spread placement & management", topic_ibkr_trading),
    ("scan-results",     "Feb 2026 S&P 500 scan findings", topic_scan_results),
    ("forward-atm",      "Put-call parity & straddle pricing", topic_forward_atm),
    ("vol-concepts",     "RV windows, IV crush, term structure", topic_vol_concepts),
    ("commands",         "Quick reference for all CLI commands", topic_commands),
]

TOPIC_MAP = {name: fn for name, _, fn in TOPICS}


def list_topics():
    header("AVAILABLE TOPICS")
    for name, desc, _ in TOPICS:
        print(f"  {GREEN}{name:<20}{RESET} {desc}")
    print(f"\n  Usage: {DIM}python study.py <topic>{RESET}\n")


def print_all():
    print(f"\n{BOLD}{CYAN}{'━' * 60}")
    print(f"  OPTIONS TRADING KNOWLEDGE BASE")
    print(f"{'━' * 60}{RESET}")
    print(f"  {DIM}17 topics • Built from S&P 500 spread scanning")
    print(f"  IBKR + WealthSimple trading • Feb 2026{RESET}\n")

    for _, _, fn in TOPICS:
        fn()


def main():
    if len(sys.argv) < 2:
        print_all()
        return

    arg = sys.argv[1]

    if arg in ("--list", "-l", "list"):
        list_topics()
    elif arg in ("--help", "-h"):
        print((__doc__ or "").strip())
    elif arg in TOPIC_MAP:
        TOPIC_MAP[arg]()
    else:
        print(f"{RED}Unknown topic: {arg}{RESET}")
        print(f"Run {DIM}python study.py --list{RESET} to see available topics.")
        sys.exit(1)


if __name__ == "__main__":
    main()
