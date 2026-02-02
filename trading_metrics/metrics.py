"""
Standardized Performance Metrics
================================

All calculations use quantstats library where possible.
Fallback to manual calculations if quantstats unavailable.

IMPORTANT: All projects should use these functions for consistency.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InsufficientDataError(ValueError):
    """Raised when there's not enough data to calculate metrics."""
    pass


class InvalidDataError(ValueError):
    """Raised when input data is invalid (NaN prices, negative values, etc.)."""
    pass

# Use quantstats for validated calculations
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    import warnings
    warnings.warn("quantstats not installed. Using manual calculations. Install with: pip install quantstats")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """Single trade result"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float


@dataclass
class TradeAnalysis:
    """Enhanced trade with vs-hold analysis for SELL→BUY cycles"""
    sell_date: str
    sell_price: float
    sell_reason: str
    buy_date: str
    buy_price: float
    buy_reason: str
    price_change_while_out: float  # What market did while we were out
    benefit: float                  # Positive = avoided loss, Negative = missed gain
    analysis: str                   # "Avoided 5% drop" or "Missed 3% gain"


@dataclass
class BaselineComparison:
    """Strategy vs Buy-and-Hold comparison"""
    strategy_return: float
    buy_hold_return: float
    outperformance: float          # strategy - buy_hold
    outperformance_pct: float      # (strategy / buy_hold - 1) * 100


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float  # Trade-based win rate
    num_trades: int
    avg_trade_return: float
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'avg_trade_return': self.avg_trade_return,
            'total_predictions': self.total_signals,
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals,
            'hold_signals': self.hold_signals
        }


@dataclass
class StrategySimulationResult:
    """
    Result from simulate_strategy_from_invested().

    PARADIGM: Start 100% invested, every cycle is SELL→BUY.
    """
    trades: List[Dict]              # List of {date, action, price, reason}
    trade_cycles: List[TradeAnalysis]  # SELL→BUY pairs with analysis
    equity: pd.Series               # Strategy equity curve
    baseline_equity: pd.Series      # Buy-and-hold equity curve (for comparison)
    realized_pnl: float             # Total realized P&L
    final_position: int             # 1=invested, 0=out (always 1 after implicit buy)


@dataclass
class BacktestResult:
    """
    Complete backtest result with metrics, equity curves, and baseline comparison.

    This is the standard result format for all backtests/model evaluations.
    Use run_backtest_with_curves() to get this result.
    """
    # Core metrics from trading-metrics
    metrics: BacktestMetrics

    # Baseline comparison
    baseline: BaselineComparison

    # Equity curves for dual-line charts (both scaled to same initial value)
    equity_curve: List[Dict]  # [{date, strategy, baseline}, ...]


# =============================================================================
# Core Metric Calculations
# =============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Formula: (mean(returns) - risk_free_rate) / std(returns) * sqrt(periods)

    Args:
        returns: Series of periodic returns (daily returns if periods=252)
        risk_free_rate: Annual risk-free rate (default 0)
        periods: Annualization factor (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    if returns is None or len(returns) < 2:
        return 0.0

    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.sharpe(returns, rf=risk_free_rate, periods=periods)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats sharpe failed, using fallback: {e}")

    # Fallback: manual calculation
    excess_returns = returns - risk_free_rate / periods
    return float(np.sqrt(periods) * excess_returns.mean() / excess_returns.std())


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation instead of std).

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods: Annualization factor

    Returns:
        Annualized Sortino ratio
    """
    if returns is None or len(returns) < 2:
        return 0.0

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.sortino(returns, rf=risk_free_rate, periods=periods)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats sortino failed, using fallback: {e}")

    # Fallback: manual calculation
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess_returns.mean() / downside_returns.std())


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from returns series.

    Args:
        returns: Series of periodic returns

    Returns:
        Max drawdown as negative decimal (e.g., -0.15 for 15% drawdown)
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.max_drawdown(returns)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats max_drawdown failed, using fallback: {e}")

    # Fallback: manual calculation
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())


def calculate_total_return(returns: pd.Series) -> float:
    """
    Calculate total cumulative return.

    Formula: product(1 + r) - 1

    Args:
        returns: Series of periodic returns

    Returns:
        Total return as decimal (e.g., 0.25 for 25%)
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.comp(returns)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats comp failed, using fallback: {e}")

    # Fallback: manual calculation
    return float((1 + returns).prod() - 1)


def calculate_cagr(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        returns: Series of periodic returns
        periods: Periods per year (252 for daily)

    Returns:
        CAGR as decimal
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.cagr(returns, periods=periods)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats cagr failed, using fallback: {e}")

    # Fallback: manual calculation
    total_return = (1 + returns).prod()
    years = len(returns) / periods
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def calculate_volatility(returns: pd.Series, periods: int = 252) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Series of periodic returns
        periods: Periods per year (252 for daily)

    Returns:
        Annualized volatility as decimal
    """
    if returns is None or len(returns) < 2:
        return 0.0

    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    if HAS_QUANTSTATS:
        try:
            result = qs.stats.volatility(returns, periods=periods)
            return float(result) if not pd.isna(result) else 0.0
        except Exception as e:
            logger.debug(f"quantstats volatility failed, using fallback: {e}")

    # Fallback: manual calculation
    return float(returns.std() * np.sqrt(periods))


# =============================================================================
# Win Rate Calculations (multiple variants)
# =============================================================================

def calculate_trade_win_rate(trades: List[Trade]) -> float:
    """
    Calculate win rate from completed trades.

    Args:
        trades: List of Trade objects

    Returns:
        Win rate as decimal (e.g., 0.6 for 60%)
    """
    if not trades:
        return 0.0

    winners = sum(1 for t in trades if t.return_pct > 0)
    return winners / len(trades)


def calculate_avg_trade_return(trades: List[Trade]) -> float:
    """
    Calculate average return per trade.

    Args:
        trades: List of Trade objects

    Returns:
        Average return as decimal
    """
    if not trades:
        return 0.0

    return sum(t.return_pct for t in trades) / len(trades)


# =============================================================================
# Trade Simulation
# =============================================================================

def _simulate_from_signals(
    df: pd.DataFrame,
    date_col: str = 'date',
    price_col: str = 'price',
    signal_col: str = 'action'
) -> Tuple[List[Trade], pd.Series]:
    """
    Internal: Simulate trades from BUY/SELL signals.

    PARADIGM: Start 100% INVESTED (baseline = buy-and-hold)
        - SELL: Exit position (go to cash)
        - BUY: Re-enter position
        - HOLD: Stay in current state
        - If ends out of market, implicit re-entry at final price

    Every trade cycle is SELL → BUY.

    Args:
        df: DataFrame with date, price, signal columns
        date_col: Name of date column
        price_col: Name of price column
        signal_col: Name of signal column ('BUY', 'SELL', 'HOLD')

    Returns:
        Tuple of (list of trades, equity series starting at 1.0)
    """
    if df.empty:
        return [], pd.Series([1.0])

    trades = []
    equity = [1.0]

    # START 100% INVESTED
    position = 1  # 1=invested, 0=out
    first_row = df.iloc[0]
    entry_price = float(first_row[price_col])
    entry_date = str(first_row[date_col])
    realized_return = 0.0  # Cumulative realized return

    for i, row in df.iterrows():
        date = str(row[date_col])
        price = float(row[price_col])
        signal = row[signal_col]

        # SELL: Exit position (if invested)
        if signal == 'SELL' and position == 1:
            ret = (price - entry_price) / entry_price
            realized_return += ret
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=price,
                return_pct=ret
            ))
            position = 0
            equity.append(1.0 + realized_return)

        # BUY: Re-enter position (if out)
        elif signal == 'BUY' and position == 0:
            position = 1
            entry_price = price
            entry_date = date
            equity.append(1.0 + realized_return)

        # HOLD or no action: update equity based on position
        else:
            if position == 1:
                # Mark-to-market: realized + unrealized
                unrealized = (price - entry_price) / entry_price
                equity.append(1.0 + realized_return + unrealized)
            else:
                # Out of market: equity stays at realized level
                equity.append(1.0 + realized_return)

    # If ended OUT of position, add implicit re-entry at final price
    # Every trade cycle must be SELL → BUY
    if position == 0 and len(df) > 0:
        final_price = float(df.iloc[-1][price_col])
        final_date = str(df.iloc[-1][date_col])
        # Implicit BUY at end (no P&L impact, just closes the cycle)
        trades.append(Trade(
            entry_date=final_date,
            exit_date=final_date,
            entry_price=final_price,
            exit_price=final_price,
            return_pct=0.0  # Implicit re-entry, no return
        ))

    return trades, pd.Series(equity)


def simulate_strategy_from_invested(
    prices: pd.DataFrame,
    exit_condition: callable,
    reentry_condition: callable,
    date_col: str = 'date',
    price_col: str = 'close',
    initial_equity: float = 10000.0
) -> StrategySimulationResult:
    """
    Simulate a trading strategy starting 100% invested.

    PARADIGM:
        - Start 100% INVESTED (baseline = buy-and-hold)
        - Every trade cycle is SELL → BUY (exit, then re-enter)
        - If ends with SELL (out of market), implicit BUY at final price

    This standardizes the assumption that being invested is the default state.
    Strategies define when to EXIT and when to RE-ENTER.

    Args:
        prices: DataFrame with date and price columns
        exit_condition: Function(row, entry_price) -> (should_exit, reason)
        reentry_condition: Function(row) -> (should_enter, reason)
        date_col: Name of date column
        price_col: Name of price column
        initial_equity: Starting equity value

    Returns:
        StrategySimulationResult with trades, trade_cycles, equity curve

    Example:
        def exit_cond(row, entry_price):
            drawdown = (row['close'] - entry_price) / entry_price
            if drawdown <= -0.08:
                return True, f"Stop loss {drawdown:.1%}"
            return False, ""

        def reentry_cond(row):
            if row['pct_from_high'] <= -0.05:
                return True, f"Dip {row['pct_from_high']:.1%}"
            return False, ""

        result = simulate_strategy_from_invested(df, exit_cond, reentry_cond)
    """
    if prices.empty or len(prices) < 2:
        return StrategySimulationResult(
            trades=[],
            trade_cycles=[],
            equity=pd.Series([initial_equity]),
            baseline_equity=pd.Series([initial_equity]),
            realized_pnl=0.0,
            final_position=1
        )

    # Calculate BASELINE (buy-and-hold) equity curve
    first_price = float(prices.iloc[0][price_col])
    baseline_equity = [initial_equity]
    for i in range(1, len(prices)):
        current_price = float(prices.iloc[i][price_col])
        baseline_return = (current_price - first_price) / first_price
        baseline_equity.append(initial_equity * (1 + baseline_return))

    # Start 100% INVESTED for strategy
    position = 1
    entry_price = first_price
    trades = []
    equity = [initial_equity]
    realized_pnl = 0.0

    for i in range(1, len(prices)):
        row = prices.iloc[i]
        current_price = float(row[price_col])
        current_date = str(row[date_col])

        # EXIT LOGIC: Check if we should sell (exit position)
        if position == 1:
            should_exit, exit_reason = exit_condition(row, entry_price)
            if should_exit:
                # Realize P&L
                trade_pnl = (current_price - entry_price) / entry_price
                realized_pnl += trade_pnl
                position = 0
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'reason': exit_reason
                })

        # RE-ENTRY LOGIC: Check if we should buy back in
        elif position == 0:
            should_enter, entry_reason = reentry_condition(row)
            if should_enter:
                position = 1
                entry_price = current_price
                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'reason': entry_reason
                })

        # Update equity
        if position == 1:
            unrealized_pnl = (current_price - entry_price) / entry_price
            total_pnl = realized_pnl + unrealized_pnl
            equity.append(initial_equity * (1 + total_pnl))
        else:
            equity.append(initial_equity * (1 + realized_pnl))

    # IMPLICIT BUY AT END: If we end out of market, assume bought back
    # Every trade cycle must be SELL → BUY
    if position == 0 and len(trades) > 0 and trades[-1]['action'] == 'SELL':
        final_row = prices.iloc[-1]
        trades.append({
            'date': str(final_row[date_col]),
            'action': 'BUY',
            'price': float(final_row[price_col]),
            'reason': 'End of period (implicit re-entry)'
        })
        position = 1  # Now back in

    # Build trade cycles (SELL→BUY pairs)
    trade_cycles = []
    i = 0
    while i < len(trades):
        if trades[i]['action'] == 'SELL':
            sell_trade = trades[i]
            # Find next BUY
            buy_trade = None
            for j in range(i + 1, len(trades)):
                if trades[j]['action'] == 'BUY':
                    buy_trade = trades[j]
                    break

            if buy_trade:
                analysis = analyze_exit_reentry(
                    sell_date=sell_trade['date'],
                    sell_price=sell_trade['price'],
                    sell_reason=sell_trade['reason'],
                    buy_date=buy_trade['date'],
                    buy_price=buy_trade['price'],
                    buy_reason=buy_trade['reason']
                )
                trade_cycles.append(analysis)
        i += 1

    return StrategySimulationResult(
        trades=trades,
        trade_cycles=trade_cycles,
        equity=pd.Series(equity),
        baseline_equity=pd.Series(baseline_equity),
        realized_pnl=realized_pnl,
        final_position=position
    )


# =============================================================================
# High-Level Backtest
# =============================================================================

def _empty_metrics() -> BacktestMetrics:
    """Create empty BacktestMetrics with all zeros."""
    return BacktestMetrics(
        total_return=0.0, cagr=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
        max_drawdown=0.0, volatility=0.0, win_rate=0.0,
        num_trades=0, avg_trade_return=0.0,
        total_signals=0, buy_signals=0, sell_signals=0, hold_signals=0
    )


def _empty_baseline() -> BaselineComparison:
    """Create empty BaselineComparison with all zeros."""
    return BaselineComparison(
        strategy_return=0.0, buy_hold_return=0.0,
        outperformance=0.0, outperformance_pct=0.0
    )


def _empty_result() -> BacktestResult:
    """Create empty BacktestResult."""
    return BacktestResult(
        metrics=_empty_metrics(),
        baseline=_empty_baseline(),
        equity_curve=[]
    )


def _build_equity_curve(
    df: pd.DataFrame,
    strategy_equity: List[float],
    baseline_equity: pd.Series,
    date_col: str
) -> List[Dict]:
    """Build equity curve for frontend dual-line chart."""
    equity_curve = []
    for i in range(len(df)):
        equity_curve.append({
            "date": str(df.iloc[i][date_col]),
            "strategy": float(strategy_equity[i]) if i < len(strategy_equity) else float(strategy_equity[-1]),
            "baseline": float(baseline_equity.iloc[i]) if i < len(baseline_equity) else float(baseline_equity.iloc[-1]),
        })
    return equity_curve


def _run_backtest_internal(
    df: pd.DataFrame,
    date_col: str = 'date',
    price_col: str = 'price',
    signal_col: str = 'action'
) -> Tuple[BacktestMetrics, List[Trade], pd.Series]:
    """
    Internal backtest that returns metrics, trades, and equity.

    Used by public functions to avoid duplicate simulation.
    """
    if df.empty:
        return _empty_metrics(), [], pd.Series([1.0])

    # Count signals
    buy_signals = int((df[signal_col] == 'BUY').sum())
    sell_signals = int((df[signal_col] == 'SELL').sum())
    hold_signals = int((df[signal_col] == 'HOLD').sum())

    # Simulate trades (only done ONCE)
    trades, equity = _simulate_from_signals(df, date_col, price_col, signal_col)

    # Calculate returns from equity curve
    returns = equity.pct_change().dropna()

    if len(returns) == 0:
        metrics = _empty_metrics()
        metrics.total_signals = len(df)
        metrics.buy_signals = buy_signals
        metrics.sell_signals = sell_signals
        metrics.hold_signals = hold_signals
        return metrics, trades, equity

    # Get dates for monthly calculation
    dates = df[date_col] if date_col in df.columns else None

    metrics = BacktestMetrics(
        total_return=calculate_total_return(returns),
        cagr=calculate_cagr(returns),
        sharpe_ratio=calculate_sharpe_ratio(returns),
        sortino_ratio=calculate_sortino_ratio(returns),
        max_drawdown=calculate_max_drawdown(returns),
        volatility=calculate_volatility(returns),
        win_rate=calculate_trade_win_rate(trades),
        num_trades=len(trades),
        avg_trade_return=calculate_avg_trade_return(trades),
        total_signals=len(df),
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        hold_signals=hold_signals
    )

    return metrics, trades, equity


def run_backtest(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    date_col: str,
    price_col: str,
    signal_col: str = 'action',
    initial_equity: float = 10000.0
) -> BacktestResult:
    """
    Run backtest on sparse BUY/SELL signals with price data for equity curves.

    PARADIGM: Start 100% invested
        - SELL: Exit position (go to cash)
        - BUY: Re-enter position
        - If period ends while out, equity stays at cash level

    Args:
        signals_df: DataFrame with sparse BUY/SELL signals (date, price, action)
                    From predictions table. Can be empty (= stayed invested).
        prices_df: DataFrame with daily prices for the period (date, close)
                   From raw_prices table. Used for equity curves and B&H calc.
        date_col: Name of date column (default 'date')
        price_col: Name of price column (default 'close')
        signal_col: Name of signal column (default 'action')
        initial_equity: Starting equity value for curves (default 10000)

    Returns:
        BacktestResult with metrics, baseline comparison, and equity curves

    Example:
        # Sparse signals from predictions table
        signals = pd.DataFrame({
            'date': ['2024-01-15', '2024-02-01'],
            'close': [110, 95],
            'action': ['SELL', 'BUY']
        })

        # Daily prices from raw_prices table
        prices = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-03-01'),
            'close': [100, 101, 102, ...]  # daily prices
        })

        result = run_backtest(signals, prices)
    """
    # Validate prices - raise clear errors instead of returning empty results
    if prices_df is None:
        raise InsufficientDataError("prices_df is required but was None")
    if prices_df.empty:
        raise InsufficientDataError("prices_df is empty - no price data provided")
    if len(prices_df) < 2:
        raise InsufficientDataError(f"prices_df has only {len(prices_df)} row(s) - need at least 2 for return calculation")
    if price_col not in prices_df.columns:
        raise InvalidDataError(f"price column '{price_col}' not found in prices_df. Available: {list(prices_df.columns)}")
    if date_col not in prices_df.columns:
        raise InvalidDataError(f"date column '{date_col}' not found in prices_df. Available: {list(prices_df.columns)}")

    prices = prices_df.copy()
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices.sort_values(date_col).reset_index(drop=True)

    # Validate price values
    if prices[price_col].isna().any():
        nan_count = prices[price_col].isna().sum()
        raise InvalidDataError(f"prices_df contains {nan_count} NaN price value(s) - clean data before backtest")

    # Get boundaries from prices
    start_price = float(prices[price_col].iloc[0])
    end_price = float(prices[price_col].iloc[-1])
    start_date = prices[date_col].iloc[0]
    end_date = prices[date_col].iloc[-1]

    if start_price <= 0:
        raise InvalidDataError(f"start price is {start_price} - prices must be positive")
    if end_price <= 0:
        raise InvalidDataError(f"end price is {end_price} - prices must be positive")

    # Calculate B&H return from price boundaries
    buy_hold_return = (end_price / start_price) - 1

    # Handle empty signals - just return B&H (stayed invested whole period)
    if signals_df is None or signals_df.empty:
        # Build baseline equity curve from prices
        baseline_equity = _calculate_baseline_equity(prices[price_col], initial_equity)
        equity_curve = [
            {
                'date': str(prices[date_col].iloc[i].date()) if hasattr(prices[date_col].iloc[i], 'date') else str(prices[date_col].iloc[i]),
                'strategy': baseline_equity.iloc[i],
                'baseline': baseline_equity.iloc[i]
            }
            for i in range(len(prices))
        ]
        baseline = BaselineComparison(
            strategy_return=buy_hold_return,
            buy_hold_return=buy_hold_return,
            outperformance=0.0,
            outperformance_pct=0.0
        )
        metrics = _empty_metrics()
        metrics.total_return = buy_hold_return
        return BacktestResult(metrics=metrics, baseline=baseline, equity_curve=equity_curve)

    # Build backtest data: merge signals with price boundaries
    signals = signals_df.copy()
    signals[date_col] = pd.to_datetime(signals[date_col])

    # Ensure signals has correct price column name (may be 'price' from predictions table)
    # Rename to match price_col if needed
    signal_price_col = 'price' if 'price' in signals.columns else price_col
    if signal_price_col != price_col and signal_price_col in signals.columns:
        signals = signals.rename(columns={signal_price_col: price_col})

    # Create boundary rows (HOLD = stay in current position)
    start_row = pd.DataFrame({
        date_col: [start_date],
        price_col: [start_price],
        signal_col: ['HOLD']
    })
    end_row = pd.DataFrame({
        date_col: [end_date],
        price_col: [end_price],
        signal_col: ['HOLD']
    })

    # Combine: start + signals + end (sparse backtest data)
    backtest_df = pd.concat([start_row, signals, end_row], ignore_index=True)
    backtest_df = backtest_df.drop_duplicates(subset=date_col, keep='last')
    backtest_df = backtest_df.sort_values(date_col).reset_index(drop=True)

    # Run backtest simulation on sparse data
    metrics, trades, strategy_equity_norm = _run_backtest_internal(backtest_df, date_col, price_col, signal_col)

    # Build equity curves using full price data
    # Merge strategy positions with daily prices
    baseline_equity = _calculate_baseline_equity(prices[price_col], initial_equity)

    # Interpolate strategy equity to daily prices
    # Create position series from signals
    position = 1  # Start invested
    positions = {}
    for _, row in backtest_df.iterrows():
        if row[signal_col] == 'SELL':
            position = 0
        elif row[signal_col] == 'BUY':
            position = 1
        positions[row[date_col]] = position

    # Forward-fill positions across all price dates
    current_position = 1
    strategy_equity = []
    entry_price = start_price
    realized_return = 0.0

    for i, row in prices.iterrows():
        date = row[date_col]
        price = float(row[price_col])

        # Check if there's a signal on this date
        if date in positions:
            new_position = positions[date]
            if current_position == 1 and new_position == 0:
                # SELL: realize gains/losses
                realized_return += (price - entry_price) / entry_price
                current_position = 0
            elif current_position == 0 and new_position == 1:
                # BUY: re-enter
                entry_price = price
                current_position = 1

        # Calculate equity
        if current_position == 1:
            unrealized = (price - entry_price) / entry_price
            equity = initial_equity * (1 + realized_return + unrealized)
        else:
            equity = initial_equity * (1 + realized_return)

        strategy_equity.append(equity)

    # Build equity curve
    equity_curve = [
        {
            'date': str(prices[date_col].iloc[i].date()) if hasattr(prices[date_col].iloc[i], 'date') else str(prices[date_col].iloc[i]),
            'strategy': strategy_equity[i],
            'baseline': baseline_equity.iloc[i]
        }
        for i in range(len(prices))
    ]

    # Calculate baseline comparison
    strategy_return = metrics.total_return
    outperformance = strategy_return - buy_hold_return
    outperformance_pct = ((1 + strategy_return) / (1 + buy_hold_return) - 1) * 100 if buy_hold_return != -1 else 0

    baseline = BaselineComparison(
        strategy_return=strategy_return,
        buy_hold_return=buy_hold_return,
        outperformance=outperformance,
        outperformance_pct=outperformance_pct
    )

    return BacktestResult(
        metrics=metrics,
        baseline=baseline,
        equity_curve=equity_curve
    )


def metrics_to_dict(metrics: BacktestMetrics) -> Dict:
    """
    Convert BacktestMetrics to dict for JSON serialization.

    DEPRECATED: Use metrics.to_dict() instead.
    """
    return metrics.to_dict()


# =============================================================================
# Baseline Comparison Functions
# =============================================================================

def calculate_buy_hold_return(prices: pd.Series = None, *, start_price: float = None, end_price: float = None) -> float:
    """
    Calculate buy-and-hold return.

    Can be called two ways:
    1. With a price series: calculate_buy_hold_return(prices)
    2. With start/end prices: calculate_buy_hold_return(start_price=100, end_price=120)

    Args:
        prices: Series of prices (first value = entry, last value = exit)
        start_price: Starting price (keyword-only)
        end_price: Ending price (keyword-only)

    Returns:
        Total return as decimal (e.g., 0.25 for 25%)
    """
    # If start_price and end_price provided, use them directly
    if start_price is not None and end_price is not None:
        if start_price <= 0:
            return 0.0
        return float((end_price / start_price) - 1)

    # Otherwise use price series
    if prices is None or len(prices) < 2:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


def _calculate_baseline_equity(
    prices: pd.Series,
    initial_equity: float = 10000.0
) -> pd.Series:
    """
    Calculate buy-and-hold equity curve for baseline comparison on charts.

    This is the standard baseline for all performance charts:
    - Start 100% invested at initial_equity
    - Track value over time as if held without any trades

    Args:
        prices: Series of prices
        initial_equity: Starting equity value (default 10000)

    Returns:
        Series of equity values (same length as prices)

    Example:
        prices = pd.Series([100, 105, 95, 110])
        baseline = _calculate_baseline_equity(prices, 10000)
        # Returns: [10000, 10500, 9500, 11000]
    """
    if prices is None or len(prices) < 1:
        return pd.Series([initial_equity])

    first_price = float(prices.iloc[0])
    if first_price == 0:
        return pd.Series([initial_equity] * len(prices))

    equity = []
    for price in prices:
        ret = (float(price) - first_price) / first_price
        equity.append(initial_equity * (1 + ret))

    return pd.Series(equity)


def analyze_exit_reentry(
    sell_date: str,
    sell_price: float,
    sell_reason: str,
    buy_date: str,
    buy_price: float,
    buy_reason: str
) -> TradeAnalysis:
    """
    Analyze a SELL→BUY cycle vs holding through.

    Baseline: 100% invested (bought in)
    Trade: SELL at sell_price, BUY back at buy_price

    If market went DOWN while out: "Avoided X% drop" (good exit)
    If market went UP while out: "Missed X% gain" (bad exit)

    Args:
        sell_date: Date of SELL
        sell_price: Price when sold
        sell_reason: Reason for selling
        buy_date: Date of re-entry BUY
        buy_price: Price when bought back
        buy_reason: Reason for buying

    Returns:
        TradeAnalysis with vs-hold comparison
    """
    # What happened to price while we were OUT?
    price_change_while_out = (buy_price - sell_price) / sell_price

    # Our benefit: negative price change = we saved money
    # If buy_price < sell_price: we saved money (positive outcome)
    # If buy_price > sell_price: we missed gains (negative outcome)
    benefit = -price_change_while_out  # Flip sign: down market = positive for us

    # Generate analysis text
    if price_change_while_out < -0.001:  # Market went DOWN
        analysis = f"Avoided {abs(price_change_while_out)*100:.1f}% drop"
    elif price_change_while_out > 0.001:  # Market went UP
        analysis = f"Missed {price_change_while_out*100:.1f}% gain"
    else:
        analysis = "Neutral"

    return TradeAnalysis(
        sell_date=sell_date,
        sell_price=sell_price,
        sell_reason=sell_reason,
        buy_date=buy_date,
        buy_price=buy_price,
        buy_reason=buy_reason,
        price_change_while_out=price_change_while_out,
        benefit=benefit,
        analysis=analysis
    )
