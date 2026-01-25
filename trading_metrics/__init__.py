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

    # Simulation
    simulate_trades,

    # High-level backtest
    run_backtest,
    BacktestMetrics,
    Trade,

    # Serialization
    metrics_to_dict,
)

__version__ = "0.1.0"
__all__ = [
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
    "run_backtest",
    "BacktestMetrics",
    "Trade",
    "metrics_to_dict",
]
