"""
Strategy Signal Generators - Dispatcher
========================================

Routes to specific strategy implementations.
For direct usage, import from the strategy module:

    from trading_metrics.stoploss import generate_stoploss_signals
    from trading_metrics.momentum import generate_momentum_signals

This module provides backward-compatible generate_signals() dispatcher.
"""

import pandas as pd
from typing import Dict

from .stoploss import generate_stoploss_signals
from .momentum import generate_momentum_signals
from .metrics import InsufficientDataError, InvalidDataError


# Re-export for backward compatibility
from .stoploss import generate_stoploss_signals as _generate_stoploss_signals
from .momentum import generate_momentum_signals as _generate_momentum_signals


def generate_signals(
    model_type: str,
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    include_hold: bool = False
) -> pd.DataFrame:
    """
    Generate trading signals for any model type.

    This is a dispatcher that routes to specific strategy implementations.
    For direct usage, prefer importing from the strategy module directly:

        from trading_metrics.stoploss import generate_stoploss_signals

    Args:
        model_type: Strategy type ("stoploss", "dip_buyer", "custom", "momentum")
        config: Strategy configuration parameters
        prices_df: DataFrame with OHLCV data
        symbol: Trading symbol
        date_col: Name of date column
        price_col: Name of close price column
        high_col: Name of high price column
        low_col: Name of low price column
        include_hold: If True, include HOLD signals for every trading day

    Returns:
        DataFrame with columns: date, action, price, reason, symbol

    Raises:
        InsufficientDataError: If prices_df is empty
        InvalidDataError: If required columns missing
        ValueError: If unknown model type
    """
    if model_type in ("stoploss", "dip_buyer", "custom"):
        return generate_stoploss_signals(
            config=config,
            prices_df=prices_df,
            symbol=symbol,
            date_col=date_col,
            price_col=price_col,
            high_col=high_col,
            include_hold=include_hold
        )
    elif model_type == "momentum":
        return generate_momentum_signals(
            config=config,
            prices_df=prices_df,
            symbol=symbol,
            date_col=date_col,
            price_col=price_col,
            include_hold=include_hold
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: stoploss, dip_buyer, momentum")


# Legacy exports for backward compatibility
def prepare_indicators(df, config, price_col='close', high_col='high'):
    """Prepare indicators for strategy simulation. DEPRECATED - use stoploss.py directly."""
    result = df.copy()
    buy_dip_lookback = int(config.get('buy_dip_lookback', 20))
    trailing_stop_lookback = int(config.get('trailing_stop_lookback', 10))

    if price_col not in result.columns:
        price_col = 'Close' if 'Close' in result.columns else price_col
    if high_col not in result.columns:
        high_col = 'High' if 'High' in result.columns else high_col

    result[f'high_{buy_dip_lookback}d'] = result[high_col].rolling(buy_dip_lookback, min_periods=1).max()
    result['pct_from_high'] = (result[price_col] - result[f'high_{buy_dip_lookback}d']) / result[f'high_{buy_dip_lookback}d']
    result['cum_max'] = result[price_col].cummax()
    result['drawdown'] = (result[price_col] - result['cum_max']) / result['cum_max']
    result[f'rolling_high_{trailing_stop_lookback}d'] = result[high_col].rolling(trailing_stop_lookback, min_periods=1).max()

    return result


def make_exit_condition(config, trailing_lookback=10):
    """Create exit condition function. DEPRECATED - use stoploss.py directly."""
    sell_drawdown_threshold = config.get('sell_drawdown_threshold', 0.08)
    sell_vix_threshold = config.get('sell_vix_threshold')
    take_profit_pct = config.get('take_profit_pct')
    stop_loss_pct = config.get('stop_loss_pct')
    trailing_stop_pct = config.get('trailing_stop_pct')

    def exit_condition(row, entry_price):
        if row['drawdown'] <= -sell_drawdown_threshold:
            return True, f"Drawdown {row['drawdown']:.2%}"
        if sell_vix_threshold and 'vix_close' in row.index and pd.notna(row.get('vix_close')) and row['vix_close'] > sell_vix_threshold:
            return True, f"VIX {row['vix_close']:.1f}"
        if take_profit_pct and entry_price:
            price_col = 'close' if 'close' in row.index else 'Close'
            profit = (row[price_col] - entry_price) / entry_price
            if profit >= take_profit_pct:
                return True, f"Take profit {profit:.2%}"
        if stop_loss_pct and entry_price:
            price_col = 'close' if 'close' in row.index else 'Close'
            loss = (row[price_col] - entry_price) / entry_price
            if loss <= -stop_loss_pct:
                return True, f"Stop loss {loss:.2%}"
        if trailing_stop_pct:
            col = f'rolling_high_{trailing_lookback}d'
            price_col = 'close' if 'close' in row.index else 'Close'
            if col in row.index and pd.notna(row.get(col)):
                pct = (row[price_col] - row[col]) / row[col]
                if pct <= -trailing_stop_pct:
                    return True, f"Trailing stop {pct:.2%}"
        return False, ""

    return exit_condition


def make_reentry_condition(config):
    """Create reentry condition function. DEPRECATED - use stoploss.py directly."""
    buy_dip_threshold = config.get('buy_dip_threshold', 0.05)

    def reentry_condition(row):
        if row['pct_from_high'] <= -buy_dip_threshold:
            return True, f"Dip {row['pct_from_high']:.2%}"
        return False, ""

    return reentry_condition


def backtest_strategy(
    model_type: str,
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    start_date=None,
    end_date=None,
    date_col: str = 'date',
    price_col: str = 'close',
    high_col: str = 'high'
):
    """
    Run full backtest: generate signals -> calculate metrics.
    """
    from .metrics import run_backtest

    df = prices_df.copy()
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("No price data in specified date range")

    signals_df = generate_signals(
        model_type=model_type,
        config=config,
        prices_df=df,
        symbol=symbol,
        date_col=date_col,
        price_col=price_col,
        high_col=high_col
    )

    backtest_prices = df[[date_col, price_col]].copy()

    result = run_backtest(
        signals_df=signals_df,
        prices_df=backtest_prices,
        date_col=date_col,
        price_col=price_col,
        signal_col='action'
    )

    return result
