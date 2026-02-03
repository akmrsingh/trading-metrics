"""
Momentum Strategy
=================

MA crossover strategy for trend following.

Usage:
    from trading_metrics.momentum import generate_momentum_signals

    signals = generate_momentum_signals(
        config={"fast_period": 10, "slow_period": 30},
        prices_df=prices,
        symbol="SPY"
    )
"""

import pandas as pd
import numpy as np
from typing import Dict
from .metrics import InsufficientDataError, InvalidDataError


def generate_momentum_signals(
    config: Dict,
    prices_df: pd.DataFrame,
    symbol: str,
    date_col: str = 'date',
    price_col: str = 'close',
    include_hold: bool = False
) -> pd.DataFrame:
    """
    Generate signals for momentum/MA crossover strategy.

    Args:
        config: Strategy configuration:
            - fast_period: int - Fast MA period (default: 10)
            - slow_period: int - Slow MA period (default: 30)
        prices_df: DataFrame with price data
        symbol: Trading symbol
        date_col: Name of date column
        price_col: Name of close price column
        include_hold: If True, include HOLD signals for days with no crossover

    Returns:
        DataFrame with columns: date, action, price, reason, symbol
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
        price = row[price_col]
        date = row[date_col]
        action = None
        reason = None

        if pd.isna(row['_signal_change']):
            # First row, no signal change yet
            if include_hold:
                action = 'HOLD'
                reason = 'Insufficient history for MA crossover'
        else:
            # Bearish crossover (fast crosses below slow) -> SELL
            if row['_signal_change'] == -1 and is_invested:
                action = 'SELL'
                reason = f"Bearish crossover (MA{fast_period} < MA{slow_period})"
                is_invested = False

            # Bullish crossover (fast crosses above slow) -> BUY
            elif row['_signal_change'] == 1 and not is_invested:
                action = 'BUY'
                reason = f"Bullish crossover (MA{fast_period} > MA{slow_period})"
                is_invested = True

            elif include_hold:
                if is_invested:
                    action = 'HOLD'
                    reason = 'Holding position (no crossover)'
                else:
                    action = 'HOLD'
                    reason = 'Waiting for bullish crossover'

        if action is not None:
            signals.append({
                date_col: date,
                'action': action,
                'price': price,
                'reason': reason,
                'symbol': symbol
            })

    if not signals:
        return pd.DataFrame(columns=[date_col, 'action', 'price', 'reason', 'symbol'])

    return pd.DataFrame(signals)
