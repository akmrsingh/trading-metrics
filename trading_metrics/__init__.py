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

from .metrics import (
    # Core metric calculations
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    calculate_cagr,
    calculate_volatility,

    # Win rate variants
    calculate_trade_win_rate,
    calculate_daily_win_rate,
    calculate_monthly_win_rate,

    # Trade helpers
    calculate_avg_trade_return,

    # Simulation (legacy - starts with cash)
    simulate_trades,

    # Simulation (standardized - starts 100% invested)
    simulate_strategy_from_invested,
    StrategySimulationResult,

    # High-level backtest
    run_backtest,
    run_backtest_with_curves,
    run_backtest_with_boundaries,
    BacktestMetrics,
    BacktestResult,
    Trade,

    # Baseline comparison
    TradeAnalysis,
    BaselineComparison,
    calculate_buy_hold_return,
    calculate_baseline_equity,
    compare_to_baseline,
    analyze_exit_reentry,

    # Serialization
    metrics_to_dict,
)

__version__ = "0.1.0"
__all__ = [
    "backfill_db",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_total_return",
    "calculate_cagr",
    "calculate_volatility",
    "calculate_trade_win_rate",
    "calculate_daily_win_rate",
    "calculate_monthly_win_rate",
    "calculate_avg_trade_return",
    "simulate_trades",
    "simulate_strategy_from_invested",
    "StrategySimulationResult",
    "run_backtest",
    "run_backtest_with_curves",
    "run_backtest_with_boundaries",
    "BacktestMetrics",
    "BacktestResult",
    "Trade",
    "TradeAnalysis",
    "BaselineComparison",
    "calculate_buy_hold_return",
    "calculate_baseline_equity",
    "compare_to_baseline",
    "analyze_exit_reentry",
    "metrics_to_dict",
]
