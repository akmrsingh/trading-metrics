"""
Stoploss Strategy - Single Source of Truth
==========================================

Dip-buying strategy with configurable exit conditions.
Used by orchestrator (inference), backfill worker, and optimizer.

Usage:
    from trading_metrics.stoploss import generate_stoploss_signals

    signals = generate_stoploss_signals(
        config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
        prices_df=prices,
        symbol="QQQ"
    )
"""

import pandas as pd
import numpy as np
from typing import Dict
from .metrics import InsufficientDataError, InvalidDataError


def generate_stoploss_signals(
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close',
    high_col: str = 'high',
    include_hold: bool = False
) -> pd.DataFrame:
    """
    Generate trading signals for stoploss/dip-buyer strategy.

    Paradigm: Start 100% invested
    - SELL when exit conditions met (drawdown, VIX, take profit, stop loss, trailing stop)
    - BUY when entry conditions met (dip from rolling high)
    - HOLD when no conditions triggered (explicit decision)

    Args:
        config: Strategy configuration parameters:
            - buy_dip_threshold: float - Buy when X% below N-day high (default: 0.05)
            - buy_dip_lookback: int - Days to look back for high (default: 20)
            - sell_drawdown_threshold: float - Sell when drawdown exceeds X% (default: 0.08)
            - sell_vix_threshold: float - Sell when VIX > X (optional)
            - take_profit_pct: float - Take profit at X% gain (optional)
            - stop_loss_pct: float - Stop loss at X% decline (optional)
            - trailing_stop_pct: float - Trailing stop X% from N-day high (optional)
            - trailing_stop_lookback: int - Days for trailing stop (default: 10)
        prices_df: DataFrame with OHLCV data
        symbol: Trading symbol (for reference in output)
        date_col: Name of date column
        price_col: Name of close price column
        high_col: Name of high price column
        include_hold: If True, include HOLD signals for every trading day.
            HOLD is a real model decision meaning "no exit/entry conditions met".

    Returns:
        DataFrame with columns: date, action, price, reason, symbol
        If include_hold=False: Only BUY and SELL rows (sparse)
        If include_hold=True: One row per trading day (BUY, SELL, or HOLD)

    Raises:
        InsufficientDataError: If prices_df is empty
        InvalidDataError: If required columns missing or contain NaN
    """
    # Validate inputs
    if prices_df is None:
        raise InsufficientDataError("prices_df is required but was None")
    if prices_df.empty:
        raise InsufficientDataError("prices_df is empty - no price data provided")
    if price_col not in prices_df.columns:
        raise InvalidDataError(f"price column '{price_col}' not found. Available: {list(prices_df.columns)}")
    if date_col not in prices_df.columns:
        raise InvalidDataError(f"date column '{date_col}' not found. Available: {list(prices_df.columns)}")
    if prices_df[price_col].isna().any():
        nan_count = prices_df[price_col].isna().sum()
        raise InvalidDataError(f"prices_df contains {nan_count} NaN price value(s)")

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
    entry_price = float(df.iloc[0][price_col])

    for idx, row in df.iterrows():
        price = row[price_col]
        date = row[date_col]
        action = None
        reason = None

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
                action = 'SELL'
                reason = sell_signal
                is_invested = False
                entry_price = None
            elif include_hold:
                action = 'HOLD'
                reason = 'Holding position'

        else:
            # Check ENTRY conditions
            if row['_pct_from_high'] <= -buy_dip_threshold:
                action = 'BUY'
                reason = f"Dip {row['_pct_from_high']:.1%}"
                is_invested = True
                entry_price = price
            elif include_hold:
                action = 'HOLD'
                reason = 'Waiting for entry'

        # Add signal if action was determined
        if action is not None:
            signals.append({
                date_col: date,
                'action': action,
                'price': price,
                'reason': reason,
                'symbol': symbol
            })

    # Return signals DataFrame
    if not signals:
        return pd.DataFrame(columns=[date_col, 'action', 'price', 'reason', 'symbol'])

    return pd.DataFrame(signals)
