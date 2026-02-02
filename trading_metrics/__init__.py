"""
Trading Metrics - Shared performance calculations
=================================================

Standardized performance metrics using quantstats library.
All trading projects should use these functions for consistency.

Usage:
    from trading_metrics import calculate_sharpe_ratio, calculate_max_drawdown

    sharpe = calculate_sharpe_ratio(returns_series)
    max_dd = calculate_max_drawdown(returns_series)
"""

from . import backfill_db
from . import strategies

from .strategies import (
    # Signal generation
    generate_signals,
    backtest_strategy,
    # Indicator preparation (for simulation)
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
    "generate_signals",
    "backtest_strategy",
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
