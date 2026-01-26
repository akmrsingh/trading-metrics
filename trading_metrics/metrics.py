"""
Standardized Performance Metrics
================================

All calculations use quantstats library where possible.
Fallback to manual calculations if quantstats unavailable.

IMPORTANT: All projects should use these functions for consistency.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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


@dataclass
class BacktestMetrics:
    """Complete backtest performance metrics"""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float  # Trade-based
    daily_win_rate: float
    monthly_win_rate: float
    num_trades: int
    avg_trade_return: float
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int


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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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


def calculate_daily_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate from daily returns.

    Args:
        returns: Series of daily returns

    Returns:
        Win rate as decimal (% of days with positive returns)
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna()
    non_zero_returns = returns[returns != 0]

    if len(non_zero_returns) == 0:
        return 0.0

    winning_days = (non_zero_returns > 0).sum()
    return float(winning_days / len(non_zero_returns))


def calculate_monthly_win_rate(returns: pd.Series, dates: Optional[pd.Series] = None) -> float:
    """
    Calculate win rate from monthly returns.

    Args:
        returns: Series of daily returns
        dates: Optional date index (if returns doesn't have datetime index)

    Returns:
        Win rate as decimal (% of months with positive returns)
    """
    if returns is None or len(returns) == 0:
        return 0.0

    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Try to get monthly returns
    try:
        if dates is not None:
            df = pd.DataFrame({'date': dates, 'return': returns})
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly = df.groupby('month')['return'].apply(lambda x: (1 + x).prod() - 1)
        elif hasattr(returns.index, 'to_period'):
            monthly = returns.groupby(returns.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
        else:
            # Can't determine monthly, fall back to daily
            return calculate_daily_win_rate(returns)

        if len(monthly) == 0:
            return 0.0

        positive_months = (monthly > 0).sum()
        return float(positive_months / len(monthly))
    except Exception:
        return calculate_daily_win_rate(returns)


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

def simulate_trades(
    df: pd.DataFrame,
    date_col: str = 'date',
    price_col: str = 'price',
    signal_col: str = 'action'
) -> Tuple[List[Trade], pd.Series]:
    """
    Simulate trades from BUY/SELL signals.

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

def run_backtest(
    df: pd.DataFrame,
    date_col: str = 'date',
    price_col: str = 'price',
    signal_col: str = 'action'
) -> BacktestMetrics:
    """
    Run complete backtest and return all metrics.

    This is the main entry point for backtesting.

    Args:
        df: DataFrame with date, price, signal columns
        date_col: Name of date column
        price_col: Name of price column
        signal_col: Name of signal column ('BUY', 'SELL', 'HOLD')

    Returns:
        BacktestMetrics with all performance metrics
    """
    empty_result = BacktestMetrics(
        total_return=0.0,
        cagr=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        volatility=0.0,
        win_rate=0.0,
        daily_win_rate=0.0,
        monthly_win_rate=0.0,
        num_trades=0,
        avg_trade_return=0.0,
        total_signals=0,
        buy_signals=0,
        sell_signals=0,
        hold_signals=0
    )

    if df.empty:
        return empty_result

    # Count signals
    buy_signals = int((df[signal_col] == 'BUY').sum())
    sell_signals = int((df[signal_col] == 'SELL').sum())
    hold_signals = int((df[signal_col] == 'HOLD').sum())

    # Simulate trades
    trades, equity = simulate_trades(df, date_col, price_col, signal_col)

    # Calculate returns from equity curve
    returns = equity.pct_change().dropna()

    if len(returns) == 0:
        empty_result.total_signals = len(df)
        empty_result.buy_signals = buy_signals
        empty_result.sell_signals = sell_signals
        empty_result.hold_signals = hold_signals
        return empty_result

    # Get dates for monthly calculation
    dates = df[date_col] if date_col in df.columns else None

    return BacktestMetrics(
        total_return=calculate_total_return(returns),
        cagr=calculate_cagr(returns),
        sharpe_ratio=calculate_sharpe_ratio(returns),
        sortino_ratio=calculate_sortino_ratio(returns),
        max_drawdown=calculate_max_drawdown(returns),
        volatility=calculate_volatility(returns),
        win_rate=calculate_trade_win_rate(trades),
        daily_win_rate=calculate_daily_win_rate(returns),
        monthly_win_rate=calculate_monthly_win_rate(returns, dates),
        num_trades=len(trades),
        avg_trade_return=calculate_avg_trade_return(trades),
        total_signals=len(df),
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        hold_signals=hold_signals
    )


def metrics_to_dict(metrics: BacktestMetrics) -> Dict:
    """Convert BacktestMetrics to dict for JSON serialization."""
    return {
        'total_return': metrics.total_return,
        'cagr': metrics.cagr,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'max_drawdown': metrics.max_drawdown,
        'volatility': metrics.volatility,
        'win_rate': metrics.win_rate,
        'daily_win_rate': metrics.daily_win_rate,
        'monthly_win_rate': metrics.monthly_win_rate,
        'num_trades': metrics.num_trades,
        'avg_trade_return': metrics.avg_trade_return,
        'total_predictions': metrics.total_signals,
        'buy_signals': metrics.buy_signals,
        'sell_signals': metrics.sell_signals,
        'hold_signals': metrics.hold_signals
    }


# =============================================================================
# Baseline Comparison Functions
# =============================================================================

def calculate_buy_hold_return(prices: pd.Series) -> float:
    """
    Calculate buy-and-hold return from price series.

    Args:
        prices: Series of prices (first value = entry, last value = exit)

    Returns:
        Total return as decimal (e.g., 0.25 for 25%)
    """
    if prices is None or len(prices) < 2:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


def calculate_baseline_equity(
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
        baseline = calculate_baseline_equity(prices, 10000)
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


def compare_to_baseline(
    strategy_return: float,
    prices: pd.Series
) -> BaselineComparison:
    """
    Compare strategy return to buy-and-hold baseline.

    Args:
        strategy_return: Strategy's total return (decimal)
        prices: Price series for the period

    Returns:
        BaselineComparison with outperformance metrics
    """
    buy_hold = calculate_buy_hold_return(prices)
    outperformance = strategy_return - buy_hold

    return BaselineComparison(
        strategy_return=strategy_return,
        buy_hold_return=buy_hold,
        outperformance=outperformance
    )


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
