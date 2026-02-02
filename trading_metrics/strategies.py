"""
Strategy Signal Generators
==========================

Unified signal generation for all strategy types.
Each strategy takes config + prices and outputs sparse BUY/SELL signals.

Usage:
    from trading_metrics.strategies import generate_signals

    signals_df = generate_signals(
        model_type="stoploss",
        config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
        prices_df=prices,
        symbol="QQQ"
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

from .metrics import InsufficientDataError, InvalidDataError


def prepare_indicators(
    df: pd.DataFrame,
    config: Dict,
    price_col: str = 'close',
    high_col: str = 'high'
) -> pd.DataFrame:
    """
    Add technical indicators to a price DataFrame for strategy simulation.

    This prepares the DataFrame with indicators needed for exit/entry conditions:
    - Rolling high for dip detection (pct_from_high)
    - Cumulative max for drawdown calculation
    - Rolling high for trailing stop

    Args:
        df: DataFrame with price data (must have date, close, high columns)
        config: Strategy config with lookback periods
        price_col: Name of close price column
        high_col: Name of high price column

    Returns:
        DataFrame with added indicator columns
    """
    result = df.copy()

    # Normalize column names
    if price_col not in result.columns:
        price_col = 'Close' if 'Close' in result.columns else price_col
    if high_col not in result.columns:
        high_col = 'High' if 'High' in result.columns else high_col

    buy_dip_lookback = int(config.get('buy_dip_lookback', 20))
    trailing_stop_lookback = int(config.get('trailing_stop_lookback', 10))

    # Rolling high for dip detection
    result[f'high_{buy_dip_lookback}d'] = result[high_col].rolling(buy_dip_lookback, min_periods=1).max()
    result['pct_from_high'] = (result[price_col] - result[f'high_{buy_dip_lookback}d']) / result[f'high_{buy_dip_lookback}d']

    # Cumulative max for drawdown
    result['cum_max'] = result[price_col].cummax()
    result['drawdown'] = (result[price_col] - result['cum_max']) / result['cum_max']

    # Rolling high for trailing stop
    result[f'rolling_high_{trailing_stop_lookback}d'] = result[high_col].rolling(trailing_stop_lookback, min_periods=1).max()

    return result


def make_exit_condition(config: Dict, trailing_lookback: int = 10):
    """
    Create an exit condition function for simulate_strategy_from_invested.

    Args:
        config: Strategy configuration with thresholds
        trailing_lookback: Lookback period for trailing stop

    Returns:
        Function(row, entry_price) -> (bool, str)
    """
    sell_drawdown_threshold = config.get('sell_drawdown_threshold', 0.08)
    sell_vix_threshold = config.get('sell_vix_threshold')
    take_profit_pct = config.get('take_profit_pct')
    stop_loss_pct = config.get('stop_loss_pct')
    trailing_stop_pct = config.get('trailing_stop_pct')

    def exit_condition(row, entry_price):
        import pandas as pd

        # Drawdown threshold
        if row['drawdown'] <= -sell_drawdown_threshold:
            return True, f"Drawdown {row['drawdown']:.2%}"

        # VIX threshold
        if sell_vix_threshold and 'vix_close' in row.index and pd.notna(row.get('vix_close')) and row['vix_close'] > sell_vix_threshold:
            return True, f"VIX {row['vix_close']:.1f}"

        # Take profit
        if take_profit_pct and entry_price:
            price_col = 'close' if 'close' in row.index else 'Close'
            profit = (row[price_col] - entry_price) / entry_price
            if profit >= take_profit_pct:
                return True, f"Take profit {profit:.2%}"

        # Stop loss
        if stop_loss_pct and entry_price:
            price_col = 'close' if 'close' in row.index else 'Close'
            loss = (row[price_col] - entry_price) / entry_price
            if loss <= -stop_loss_pct:
                return True, f"Stop loss {loss:.2%}"

        # Trailing stop
        if trailing_stop_pct:
            col = f'rolling_high_{trailing_lookback}d'
            price_col = 'close' if 'close' in row.index else 'Close'
            if col in row.index and pd.notna(row.get(col)):
                pct = (row[price_col] - row[col]) / row[col]
                if pct <= -trailing_stop_pct:
                    return True, f"Trailing stop {pct:.2%}"

        return False, ""

    return exit_condition


def make_reentry_condition(config: Dict):
    """
    Create a reentry condition function for simulate_strategy_from_invested.

    Args:
        config: Strategy configuration with thresholds

    Returns:
        Function(row) -> (bool, str)
    """
    buy_dip_threshold = config.get('buy_dip_threshold', 0.05)

    def reentry_condition(row):
        if row['pct_from_high'] <= -buy_dip_threshold:
            return True, f"Dip {row['pct_from_high']:.2%}"
        return False, ""

    return reentry_condition


@dataclass
class SignalResult:
    """Result from signal generation for a single day."""
    action: str  # "BUY", "SELL", or "HOLD"
    reason: str
    confidence: float = 0.5


def generate_signals(
    model_type: str,
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low'
) -> pd.DataFrame:
    """
    Generate sparse BUY/SELL signals for any model type.

    Args:
        model_type: Strategy type ("stoploss", "dip_buyer", "momentum")
        config: Strategy configuration parameters
        prices_df: DataFrame with OHLCV data
        symbol: Trading symbol (for reference in output)
        date_col: Name of date column
        price_col: Name of close price column
        high_col: Name of high price column
        low_col: Name of low price column

    Returns:
        DataFrame with columns: date, action, price, reason, symbol
        Only contains BUY and SELL rows (no HOLD)

    Raises:
        InsufficientDataError: If prices_df is empty or too short
        InvalidDataError: If required columns are missing or contain invalid values
    """
    # Validate inputs
    if prices_df is None:
        raise InsufficientDataError("prices_df is required but was None")
    if prices_df.empty:
        raise InsufficientDataError("prices_df is empty - no price data provided")
    if price_col not in prices_df.columns:
        raise InvalidDataError(f"price column '{price_col}' not found in prices_df. Available: {list(prices_df.columns)}")
    if date_col not in prices_df.columns:
        raise InvalidDataError(f"date column '{date_col}' not found in prices_df. Available: {list(prices_df.columns)}")
    if prices_df[price_col].isna().any():
        nan_count = prices_df[price_col].isna().sum()
        raise InvalidDataError(f"prices_df contains {nan_count} NaN price value(s) in '{price_col}' column")

    if model_type in ("stoploss", "dip_buyer", "custom"):
        return _generate_stoploss_signals(config, prices_df, symbol, date_col, price_col, high_col)
    elif model_type == "momentum":
        return _generate_momentum_signals(config, prices_df, symbol, date_col, price_col)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: stoploss, dip_buyer, momentum")


def _generate_stoploss_signals(
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close',
    high_col: str = 'high'
) -> pd.DataFrame:
    """
    Generate signals for stoploss/dip_buyer strategy.

    Paradigm: Start 100% invested
    - SELL when exit conditions met (drawdown, VIX, take profit, stop loss, trailing stop)
    - BUY when entry conditions met (dip from rolling high)

    Config parameters:
        buy_dip_threshold: float - Buy when X% below N-day high (default: 0.05)
        buy_dip_lookback: int - Days to look back for high (default: 20)
        sell_drawdown_threshold: float - Sell when drawdown exceeds X% (default: 0.08)
        sell_vix_threshold: float - Sell when VIX > X (optional)
        take_profit_pct: float - Take profit at X% gain (optional)
        stop_loss_pct: float - Stop loss at X% decline (optional)
        trailing_stop_pct: float - Trailing stop X% from N-day high (optional)
        trailing_stop_lookback: int - Days for trailing stop (default: 10)
    """
    df = prices_df.copy()

    # Extract config with defaults
    buy_dip_threshold = config.get('buy_dip_threshold', 0.05)
    buy_dip_lookback = config.get('buy_dip_lookback', 20)
    sell_drawdown_threshold = config.get('sell_drawdown_threshold', 0.08)
    sell_vix_threshold = config.get('sell_vix_threshold')
    take_profit_pct = config.get('take_profit_pct')
    stop_loss_pct = config.get('stop_loss_pct')
    trailing_stop_pct = config.get('trailing_stop_pct')
    trailing_stop_lookback = config.get('trailing_stop_lookback', 10)

    # Normalize column names
    if price_col not in df.columns:
        price_col = 'Close' if 'Close' in df.columns else price_col
    if high_col not in df.columns:
        high_col = 'High' if 'High' in df.columns else high_col
    if date_col not in df.columns:
        date_col = 'Date' if 'Date' in df.columns else date_col

    # Calculate indicators
    df['_rolling_high'] = df[high_col].rolling(buy_dip_lookback, min_periods=1).max()
    df['_pct_from_high'] = (df[price_col] - df['_rolling_high']) / df['_rolling_high']
    df['_cum_max'] = df[price_col].cummax()
    df['_drawdown'] = (df[price_col] - df['_cum_max']) / df['_cum_max']

    if trailing_stop_pct:
        df['_trailing_high'] = df[high_col].rolling(trailing_stop_lookback, min_periods=1).max()
        df['_pct_from_trailing'] = (df[price_col] - df['_trailing_high']) / df['_trailing_high']

    # Check for VIX column
    vix_col = None
    for col in ['vix_close', 'VIX', 'vix']:
        if col in df.columns:
            vix_col = col
            break

    # Generate signals
    signals = []
    is_invested = True  # Start 100% invested
    entry_price = float(df.iloc[0][price_col])  # Entry price is first day's price

    for idx, row in df.iterrows():
        price = row[price_col]
        date = row[date_col]

        if is_invested:
            # Check EXIT conditions
            sell_signal = None

            # 1. Drawdown threshold
            if row['_drawdown'] <= -sell_drawdown_threshold:
                sell_signal = f"Drawdown {row['_drawdown']:.1%}"

            # 2. VIX threshold
            elif sell_vix_threshold and vix_col and pd.notna(row.get(vix_col)) and row[vix_col] > sell_vix_threshold:
                sell_signal = f"VIX {row[vix_col]:.1f}"

            # 3. Take profit
            elif take_profit_pct and entry_price:
                gain = (price - entry_price) / entry_price
                if gain >= take_profit_pct:
                    sell_signal = f"Take profit {gain:.1%}"

            # 4. Stop loss
            if not sell_signal and stop_loss_pct and entry_price:
                loss = (price - entry_price) / entry_price
                if loss <= -stop_loss_pct:
                    sell_signal = f"Stop loss {loss:.1%}"

            # 5. Trailing stop
            if not sell_signal and trailing_stop_pct and '_pct_from_trailing' in df.columns:
                if row['_pct_from_trailing'] <= -trailing_stop_pct:
                    sell_signal = f"Trailing stop {row['_pct_from_trailing']:.1%}"

            if sell_signal:
                signals.append({
                    date_col: date,
                    'action': 'SELL',
                    'price': price,
                    'reason': sell_signal,
                    'symbol': symbol
                })
                is_invested = False
                entry_price = None

        else:
            # Check ENTRY conditions
            if row['_pct_from_high'] <= -buy_dip_threshold:
                signals.append({
                    date_col: date,
                    'action': 'BUY',
                    'price': price,
                    'reason': f"Dip {row['_pct_from_high']:.1%}",
                    'symbol': symbol
                })
                is_invested = True
                entry_price = price

    # Return sparse signals DataFrame
    if not signals:
        return pd.DataFrame(columns=[date_col, 'action', 'price', 'reason', 'symbol'])

    return pd.DataFrame(signals)


def _generate_momentum_signals(
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Generate signals for momentum/MA crossover strategy.

    Config parameters:
        fast_period: int - Fast MA period (default: 10)
        slow_period: int - Slow MA period (default: 30)
    """
    df = prices_df.copy()

    fast_period = config.get('fast_period', 10)
    slow_period = config.get('slow_period', 30)

    # Normalize column names
    if price_col not in df.columns:
        price_col = 'Close' if 'Close' in df.columns else price_col
    if date_col not in df.columns:
        date_col = 'Date' if 'Date' in df.columns else date_col

    # Calculate MAs
    df['_fast_ma'] = df[price_col].rolling(fast_period, min_periods=1).mean()
    df['_slow_ma'] = df[price_col].rolling(slow_period, min_periods=1).mean()
    df['_signal'] = (df['_fast_ma'] > df['_slow_ma']).astype(int)
    df['_signal_change'] = df['_signal'].diff()

    # Generate signals on crossovers
    signals = []
    is_invested = True  # Start invested

    for idx, row in df.iterrows():
        if pd.isna(row['_signal_change']):
            continue

        price = row[price_col]
        date = row[date_col]

        # Bearish crossover (fast crosses below slow) -> SELL
        if row['_signal_change'] == -1 and is_invested:
            signals.append({
                date_col: date,
                'action': 'SELL',
                'price': price,
                'reason': f"Bearish crossover (MA{fast_period} < MA{slow_period})",
                'symbol': symbol
            })
            is_invested = False

        # Bullish crossover (fast crosses above slow) -> BUY
        elif row['_signal_change'] == 1 and not is_invested:
            signals.append({
                date_col: date,
                'action': 'BUY',
                'price': price,
                'reason': f"Bullish crossover (MA{fast_period} > MA{slow_period})",
                'symbol': symbol
            })
            is_invested = True

    if not signals:
        return pd.DataFrame(columns=[date_col, 'action', 'price', 'reason', 'symbol'])

    return pd.DataFrame(signals)


# Convenience function to run full backtest
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

    Returns:
        BacktestResult with metrics, baseline comparison, and equity curve
    """
    from .metrics import run_backtest

    # Filter by date range if specified
    df = prices_df.copy()
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("No price data in specified date range")

    # Generate signals
    signals_df = generate_signals(
        model_type=model_type,
        config=config,
        prices_df=df,
        symbol=symbol,
        date_col=date_col,
        price_col=price_col,
        high_col=high_col
    )

    # Create prices DataFrame for backtest
    backtest_prices = df[[date_col, price_col]].copy()

    # Run backtest with signals and prices
    result = run_backtest(
        signals_df=signals_df,
        prices_df=backtest_prices,
        date_col=date_col,
        price_col=price_col,
        signal_col='action'
    )

    return result
