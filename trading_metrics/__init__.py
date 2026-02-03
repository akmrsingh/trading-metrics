"""
Trading Metrics - Shared performance calculations
=================================================

Standardized performance metrics using quantstats library.
All trading projects should use these functions for consistency.

Usage:
    # Direct import for stoploss (recommended)
    from trading_metrics.stoploss import generate_stoploss_signals

    # Or use the dispatcher
    from trading_metrics import generate_signals

    signals = generate_stoploss_signals(config, prices_df, symbol)
"""

from . import backfill_db
from . import strategies
from . import stoploss
from . import momentum

# Direct strategy imports (preferred)
from .stoploss import generate_stoploss_signals
from .momentum import generate_momentum_signals

# Dispatcher (backward compatible)
from .strategies import (
    generate_signals,
    backtest_strategy,
    # Legacy - deprecated, use stoploss.py directly
    prepare_indicators,
    make_exit_condition,
    make_reentry_condition,
)

from .metrics import (
    # Errors - raise these instead of returning empty results
    InsufficientDataError,
    InvalidDataError,

    # Core metric calculations
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    calculate_cagr,
    calculate_volatility,

    # Win rate
    calculate_trade_win_rate,

    # Trade helpers
    calculate_avg_trade_return,

    # Simulation (starts 100% invested, uses exit/reentry conditions)
    simulate_strategy_from_invested,
    StrategySimulationResult,

    # Backtest
    run_backtest,
    BacktestMetrics,
    BacktestResult,
    Trade,

    # Baseline comparison
    TradeAnalysis,
    BaselineComparison,
    calculate_buy_hold_return,
    analyze_exit_reentry,

    # Serialization
    metrics_to_dict,
)

__version__ = "0.1.0"
__all__ = [
    # Errors
    "InsufficientDataError",
    "InvalidDataError",
    # Modules
    "backfill_db",
    "strategies",
    "stoploss",
    "momentum",
    # Direct strategy functions (preferred)
    "generate_stoploss_signals",
    "generate_momentum_signals",
    # Dispatcher (backward compatible)
    "generate_signals",
    "backtest_strategy",
    # Legacy - deprecated
    "prepare_indicators",
    "make_exit_condition",
    "make_reentry_condition",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_total_return",
    "calculate_cagr",
    "calculate_volatility",
    "calculate_trade_win_rate",
    "calculate_avg_trade_return",
    "simulate_strategy_from_invested",
    "StrategySimulationResult",
    "run_backtest",
    "BacktestMetrics",
    "BacktestResult",
    "Trade",
    "TradeAnalysis",
    "BaselineComparison",
    "calculate_buy_hold_return",
    "analyze_exit_reentry",
    "metrics_to_dict",
]
