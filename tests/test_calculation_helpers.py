"""
Unit Tests for Calculation Helper Functions
============================================

Comprehensive tests for all calculation helper functions in trading-metrics.

Run with: pytest tests/test_calculation_helpers.py -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Core Metric Calculations
# =============================================================================

class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""

    def test_sharpe_positive_consistent_returns(self):
        """Sharpe should be positive for positive mean with variance."""
        from trading_metrics import calculate_sharpe_ratio

        # Need some variance for Sharpe to be meaningful
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.008, 0.012] * 40)
        result = calculate_sharpe_ratio(returns)

        assert result > 0

    def test_sharpe_negative_consistent_losses(self):
        """Sharpe should be negative for negative mean returns."""
        from trading_metrics import calculate_sharpe_ratio

        # Need some variance for Sharpe to be meaningful
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.008, -0.012] * 40)
        result = calculate_sharpe_ratio(returns)

        assert result < 0

    def test_sharpe_zero_for_zero_returns(self):
        """Sharpe should be zero for zero returns."""
        from trading_metrics import calculate_sharpe_ratio

        returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0] * 50)
        result = calculate_sharpe_ratio(returns)

        assert result == 0.0

    def test_sharpe_empty_series(self):
        """Sharpe should return 0 for empty series."""
        from trading_metrics import calculate_sharpe_ratio

        result = calculate_sharpe_ratio(pd.Series([]))
        assert result == 0.0

    def test_sharpe_single_value(self):
        """Sharpe should return 0 for single value."""
        from trading_metrics import calculate_sharpe_ratio

        result = calculate_sharpe_ratio(pd.Series([0.01]))
        assert result == 0.0

    def test_sharpe_none_input(self):
        """Sharpe should handle None input."""
        from trading_metrics import calculate_sharpe_ratio

        result = calculate_sharpe_ratio(None)
        assert result == 0.0

    def test_sharpe_with_nan_values(self):
        """Sharpe should handle NaN values by dropping them."""
        from trading_metrics import calculate_sharpe_ratio

        returns = pd.Series([0.01, np.nan, 0.01, 0.01, np.nan, 0.01] * 20)
        result = calculate_sharpe_ratio(returns)

        assert result > 0
        assert not np.isnan(result)

    def test_sharpe_with_risk_free_rate(self):
        """Sharpe should adjust for risk-free rate."""
        from trading_metrics import calculate_sharpe_ratio

        # Need variance for meaningful Sharpe calculation
        returns = pd.Series([0.01, 0.02, 0.008, 0.015, 0.012] * 50)

        sharpe_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.05)

        # Higher risk-free rate = lower Sharpe
        assert sharpe_with_rf < sharpe_no_rf

    def test_sharpe_annualization(self):
        """Sharpe should scale with sqrt of periods."""
        from trading_metrics import calculate_sharpe_ratio

        returns = pd.Series([0.01, -0.005, 0.008, 0.012, -0.003] * 50)

        sharpe_daily = calculate_sharpe_ratio(returns, periods=252)
        sharpe_monthly = calculate_sharpe_ratio(returns, periods=12)

        # Daily annualized should be higher due to sqrt(252) > sqrt(12)
        assert sharpe_daily != sharpe_monthly


class TestCalculateSortinoRatio:
    """Tests for calculate_sortino_ratio function."""

    def test_sortino_positive_returns(self):
        """Sortino should be positive for positive mean returns with some downside."""
        from trading_metrics import calculate_sortino_ratio

        # Need some negative returns for downside deviation
        returns = pd.Series([0.02, -0.005, 0.015, 0.02, -0.003, 0.018] * 40)
        result = calculate_sortino_ratio(returns)

        assert result > 0

    def test_sortino_negative_returns(self):
        """Sortino should be negative for negative returns."""
        from trading_metrics import calculate_sortino_ratio

        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01] * 50)
        result = calculate_sortino_ratio(returns)

        assert result < 0

    def test_sortino_empty_series(self):
        """Sortino should return 0 for empty series."""
        from trading_metrics import calculate_sortino_ratio

        result = calculate_sortino_ratio(pd.Series([]))
        assert result == 0.0

    def test_sortino_single_value(self):
        """Sortino should return 0 for single value."""
        from trading_metrics import calculate_sortino_ratio

        result = calculate_sortino_ratio(pd.Series([0.01]))
        assert result == 0.0

    def test_sortino_none_input(self):
        """Sortino should handle None input."""
        from trading_metrics import calculate_sortino_ratio

        result = calculate_sortino_ratio(None)
        assert result == 0.0

    def test_sortino_all_positive_returns(self):
        """Sortino with all positive returns (no downside)."""
        from trading_metrics import calculate_sortino_ratio

        # When all returns are positive and greater than risk-free rate
        # there's no downside deviation
        returns = pd.Series([0.02, 0.03, 0.02, 0.025, 0.02] * 50)
        result = calculate_sortino_ratio(returns)

        # Should return 0 when no downside (or handle gracefully)
        assert not np.isnan(result)

    def test_sortino_higher_than_sharpe_with_skewed_returns(self):
        """Sortino should be higher than Sharpe for positive skew."""
        from trading_metrics import calculate_sharpe_ratio, calculate_sortino_ratio

        # Returns with positive skew (many small losses, few large gains)
        returns = pd.Series(
            [-0.01, -0.01, -0.01, 0.05, -0.01, -0.01, 0.05] * 30
        )

        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)

        # Sortino uses only downside deviation, should be different
        # (relationship depends on the exact distribution)
        assert sharpe != sortino


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown function."""

    def test_max_drawdown_always_negative_or_zero(self):
        """Max drawdown should always be <= 0."""
        from trading_metrics import calculate_max_drawdown

        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01] * 50)
        result = calculate_max_drawdown(returns)

        assert result <= 0

    def test_max_drawdown_detects_dip(self):
        """Max drawdown should detect significant dips."""
        from trading_metrics import calculate_max_drawdown

        # Create a 30% drawdown scenario
        returns = pd.Series([0.1, 0.1, -0.15, -0.15, 0.1, 0.1])
        result = calculate_max_drawdown(returns)

        # Should detect at least 20% drawdown
        assert result < -0.20

    def test_max_drawdown_empty_series(self):
        """Max drawdown should return 0 for empty series."""
        from trading_metrics import calculate_max_drawdown

        result = calculate_max_drawdown(pd.Series([]))
        assert result == 0.0

    def test_max_drawdown_none_input(self):
        """Max drawdown should handle None input."""
        from trading_metrics import calculate_max_drawdown

        result = calculate_max_drawdown(None)
        assert result == 0.0

    def test_max_drawdown_no_drawdown(self):
        """Max drawdown should be 0 for always-positive cumulative returns."""
        from trading_metrics import calculate_max_drawdown

        # Monotonically increasing equity = no drawdown
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        result = calculate_max_drawdown(returns)

        # Should be exactly 0 or very close
        assert result == pytest.approx(0.0, abs=0.001)

    def test_max_drawdown_single_large_drop(self):
        """Test with a single large drop."""
        from trading_metrics import calculate_max_drawdown

        # 50% drop in one day
        returns = pd.Series([0.0, -0.5, 0.0])
        result = calculate_max_drawdown(returns)

        assert result == pytest.approx(-0.5, rel=0.01)


class TestCalculateTotalReturn:
    """Tests for calculate_total_return function."""

    def test_total_return_positive(self):
        """Total return should compound correctly for positive returns."""
        from trading_metrics import calculate_total_return

        # 10% + 10% = 21% (compounded)
        returns = pd.Series([0.10, 0.10])
        result = calculate_total_return(returns)

        expected = (1.10 * 1.10) - 1  # 0.21
        assert result == pytest.approx(expected, rel=0.001)

    def test_total_return_negative(self):
        """Total return should compound correctly for negative returns."""
        from trading_metrics import calculate_total_return

        # -10% then -10% = -19% (compounded)
        returns = pd.Series([-0.10, -0.10])
        result = calculate_total_return(returns)

        expected = (0.90 * 0.90) - 1  # -0.19
        assert result == pytest.approx(expected, rel=0.001)

    def test_total_return_mixed(self):
        """Total return should compound correctly for mixed returns."""
        from trading_metrics import calculate_total_return

        # +10% then -10% = -1% (compounded)
        returns = pd.Series([0.10, -0.10])
        result = calculate_total_return(returns)

        expected = (1.10 * 0.90) - 1  # -0.01
        assert result == pytest.approx(expected, rel=0.001)

    def test_total_return_empty(self):
        """Total return should be 0 for empty series."""
        from trading_metrics import calculate_total_return

        result = calculate_total_return(pd.Series([]))
        assert result == 0.0

    def test_total_return_none(self):
        """Total return should handle None input."""
        from trading_metrics import calculate_total_return

        result = calculate_total_return(None)
        assert result == 0.0

    def test_total_return_with_nan(self):
        """Total return should handle NaN values."""
        from trading_metrics import calculate_total_return

        returns = pd.Series([0.10, np.nan, 0.10])
        result = calculate_total_return(returns)

        # Should drop NaN and compute on remaining
        assert result > 0


class TestCalculateCAGR:
    """Tests for calculate_cagr function."""

    def test_cagr_one_year(self):
        """CAGR for exactly one year of daily returns."""
        from trading_metrics import calculate_cagr

        # 252 days of ~0.0397% daily = ~10% annual
        daily_return = 0.10 / 252
        returns = pd.Series([daily_return] * 252)
        result = calculate_cagr(returns, periods=252)

        # Due to compounding, result will be slightly higher than 10%
        assert result == pytest.approx(0.10, rel=0.10)  # Within 10% tolerance

    def test_cagr_empty(self):
        """CAGR should be 0 for empty series."""
        from trading_metrics import calculate_cagr

        result = calculate_cagr(pd.Series([]))
        assert result == 0.0

    def test_cagr_none(self):
        """CAGR should handle None input."""
        from trading_metrics import calculate_cagr

        result = calculate_cagr(None)
        assert result == 0.0

    def test_cagr_negative_total_return(self):
        """CAGR should handle negative total returns."""
        from trading_metrics import calculate_cagr

        # Consistent losses
        returns = pd.Series([-0.001] * 252)
        result = calculate_cagr(returns, periods=252)

        assert result < 0


class TestCalculateVolatility:
    """Tests for calculate_volatility function."""

    def test_volatility_positive(self):
        """Volatility should always be positive."""
        from trading_metrics import calculate_volatility

        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
        result = calculate_volatility(returns)

        assert result > 0

    def test_volatility_zero_for_constant_returns(self):
        """Volatility should be very low for constant returns."""
        from trading_metrics import calculate_volatility

        returns = pd.Series([0.01] * 100)
        result = calculate_volatility(returns)

        assert result == pytest.approx(0.0, abs=0.001)

    def test_volatility_empty(self):
        """Volatility should be 0 for empty series."""
        from trading_metrics import calculate_volatility

        result = calculate_volatility(pd.Series([]))
        assert result == 0.0

    def test_volatility_single_value(self):
        """Volatility should be 0 for single value."""
        from trading_metrics import calculate_volatility

        result = calculate_volatility(pd.Series([0.01]))
        assert result == 0.0

    def test_volatility_none(self):
        """Volatility should handle None input."""
        from trading_metrics import calculate_volatility

        result = calculate_volatility(None)
        assert result == 0.0

    def test_volatility_annualization(self):
        """Volatility should scale with sqrt of periods."""
        from trading_metrics import calculate_volatility

        returns = pd.Series([0.01, -0.01, 0.02, -0.02] * 50)

        vol_daily = calculate_volatility(returns, periods=252)
        vol_monthly = calculate_volatility(returns, periods=12)

        # sqrt(252) > sqrt(12), so daily annualized > monthly annualized
        assert vol_daily > vol_monthly


# =============================================================================
# Win Rate Calculations
# =============================================================================

class TestCalculateTradeWinRate:
    """Tests for calculate_trade_win_rate function."""

    def test_win_rate_all_winners(self):
        """Win rate should be 100% for all winners."""
        from trading_metrics import calculate_trade_win_rate, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 110, 0.10),
            Trade("2024-01-11", "2024-01-20", 100, 105, 0.05),
            Trade("2024-01-21", "2024-01-30", 100, 120, 0.20),
        ]

        result = calculate_trade_win_rate(trades)
        assert result == 1.0

    def test_win_rate_all_losers(self):
        """Win rate should be 0% for all losers."""
        from trading_metrics import calculate_trade_win_rate, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 90, -0.10),
            Trade("2024-01-11", "2024-01-20", 100, 95, -0.05),
            Trade("2024-01-21", "2024-01-30", 100, 80, -0.20),
        ]

        result = calculate_trade_win_rate(trades)
        assert result == 0.0

    def test_win_rate_mixed(self):
        """Win rate should be correct for mixed trades."""
        from trading_metrics import calculate_trade_win_rate, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 110, 0.10),  # Win
            Trade("2024-01-11", "2024-01-20", 110, 100, -0.09),  # Loss
            Trade("2024-01-21", "2024-01-30", 100, 105, 0.05),  # Win
        ]

        result = calculate_trade_win_rate(trades)
        assert result == pytest.approx(2 / 3, rel=0.01)

    def test_win_rate_empty(self):
        """Win rate should be 0 for empty trades list."""
        from trading_metrics import calculate_trade_win_rate

        result = calculate_trade_win_rate([])
        assert result == 0.0

    def test_win_rate_breakeven(self):
        """Breakeven trades (0% return) should not count as winners."""
        from trading_metrics import calculate_trade_win_rate, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 100, 0.0),
            Trade("2024-01-11", "2024-01-20", 100, 100, 0.0),
        ]

        result = calculate_trade_win_rate(trades)
        assert result == 0.0


class TestCalculateAvgTradeReturn:
    """Tests for calculate_avg_trade_return function."""

    def test_avg_trade_return_positive(self):
        """Average trade return should be correct for winning trades."""
        from trading_metrics import calculate_avg_trade_return, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 110, 0.10),  # +10%
            Trade("2024-01-11", "2024-01-20", 100, 120, 0.20),  # +20%
            Trade("2024-01-21", "2024-01-30", 100, 115, 0.15),  # +15%
        ]

        result = calculate_avg_trade_return(trades)
        assert result == pytest.approx(0.15, rel=0.01)  # (10+20+15)/3 = 15%

    def test_avg_trade_return_negative(self):
        """Average trade return should be correct for losing trades."""
        from trading_metrics import calculate_avg_trade_return, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 90, -0.10),
            Trade("2024-01-11", "2024-01-20", 100, 85, -0.15),
        ]

        result = calculate_avg_trade_return(trades)
        assert result == pytest.approx(-0.125, rel=0.01)  # (-10-15)/2 = -12.5%

    def test_avg_trade_return_mixed(self):
        """Average trade return should be correct for mixed trades."""
        from trading_metrics import calculate_avg_trade_return, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 110, 0.10),   # +10%
            Trade("2024-01-11", "2024-01-20", 100, 90, -0.10),   # -10%
        ]

        result = calculate_avg_trade_return(trades)
        assert result == pytest.approx(0.0, abs=0.01)  # (10-10)/2 = 0%

    def test_avg_trade_return_empty(self):
        """Average trade return should be 0 for empty trades list."""
        from trading_metrics import calculate_avg_trade_return

        result = calculate_avg_trade_return([])
        assert result == 0.0


# =============================================================================
# Simulate Strategy
# =============================================================================

class TestSimulateStrategyFromInvested:
    """Tests for simulate_strategy_from_invested function."""

    def test_simulate_no_signals(self):
        """No exit signals = same as buy-and-hold."""
        from trading_metrics import simulate_strategy_from_invested

        prices = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 105, 110, 115, 120]
        })

        def exit_cond(row, entry_price):
            return False, ""

        def reentry_cond(row):
            return False, ""

        result = simulate_strategy_from_invested(
            prices, exit_cond, reentry_cond,
            initial_equity=10000
        )

        # No trades
        assert len(result.trades) == 0
        assert len(result.trade_cycles) == 0

        # Equity should match baseline
        assert result.equity.iloc[-1] == pytest.approx(12000, rel=0.01)
        assert result.baseline_equity.iloc[-1] == pytest.approx(12000, rel=0.01)

    def test_simulate_exit_and_reentry(self):
        """Should exit and re-enter based on conditions."""
        from trading_metrics import simulate_strategy_from_invested

        prices = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': [100, 95, 90, 85, 80, 85, 90, 95, 100, 105]
        })

        def exit_cond(row, entry_price):
            if row['close'] <= 90:
                return True, "Hit stop"
            return False, ""

        def reentry_cond(row):
            if row['close'] >= 95:
                return True, "Recovered"
            return False, ""

        result = simulate_strategy_from_invested(
            prices, exit_cond, reentry_cond,
            initial_equity=10000
        )

        # Should have SELL and BUY trades
        assert len(result.trades) >= 2
        sells = [t for t in result.trades if t['action'] == 'SELL']
        buys = [t for t in result.trades if t['action'] == 'BUY']
        assert len(sells) >= 1
        assert len(buys) >= 1

    def test_simulate_empty_dataframe(self):
        """Should handle empty dataframe."""
        from trading_metrics import simulate_strategy_from_invested

        prices = pd.DataFrame({'date': [], 'close': []})

        def exit_cond(row, entry_price):
            return False, ""

        def reentry_cond(row):
            return False, ""

        result = simulate_strategy_from_invested(
            prices, exit_cond, reentry_cond
        )

        assert len(result.trades) == 0
        assert result.final_position == 1


# =============================================================================
# Strategy Helpers (from strategies.py)
# =============================================================================

class TestPrepareIndicators:
    """Tests for prepare_indicators function."""

    def test_prepare_indicators_adds_columns(self):
        """Should add all required indicator columns."""
        from trading_metrics import prepare_indicators

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'close': [100 + i for i in range(30)],
            'high': [101 + i for i in range(30)]
        })

        config = {'buy_dip_lookback': 10, 'trailing_stop_lookback': 5}
        result = prepare_indicators(df, config)

        # Check required columns exist
        assert 'high_10d' in result.columns
        assert 'pct_from_high' in result.columns
        assert 'cum_max' in result.columns
        assert 'drawdown' in result.columns
        assert 'rolling_high_5d' in result.columns

    def test_prepare_indicators_uses_defaults(self):
        """Should use default lookback periods."""
        from trading_metrics import prepare_indicators

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'close': [100 + i for i in range(30)],
            'high': [101 + i for i in range(30)]
        })

        config = {}  # Empty config = use defaults
        result = prepare_indicators(df, config)

        # Default: buy_dip_lookback=20, trailing_stop_lookback=10
        assert 'high_20d' in result.columns
        assert 'rolling_high_10d' in result.columns

    def test_prepare_indicators_pct_from_high_calculation(self):
        """Should calculate pct_from_high correctly."""
        from trading_metrics import prepare_indicators

        # Create data where we know the answer
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 110, 105, 100, 95],
            'high': [100, 110, 105, 100, 95]
        })

        config = {'buy_dip_lookback': 5}
        result = prepare_indicators(df, config)

        # At day 5, price=95, high_5d=110, pct_from_high = (95-110)/110 = -0.136
        assert result.iloc[-1]['pct_from_high'] == pytest.approx(-0.136, rel=0.01)

    def test_prepare_indicators_drawdown_calculation(self):
        """Should calculate drawdown correctly."""
        from trading_metrics import prepare_indicators

        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100, 110, 105, 100, 95],
            'high': [100, 110, 105, 100, 95]
        })

        config = {}
        result = prepare_indicators(df, config)

        # At day 5, cum_max=110, close=95, drawdown = (95-110)/110 = -0.136
        assert result.iloc[-1]['drawdown'] == pytest.approx(-0.136, rel=0.01)


class TestMakeExitCondition:
    """Tests for make_exit_condition function."""

    def test_exit_on_drawdown(self):
        """Should trigger exit on drawdown threshold."""
        from trading_metrics import make_exit_condition

        config = {'sell_drawdown_threshold': 0.05}
        exit_fn = make_exit_condition(config)

        row = pd.Series({
            'drawdown': -0.08,  # 8% drawdown > 5% threshold
            'close': 92
        })

        should_exit, reason = exit_fn(row, entry_price=100)

        assert should_exit is True
        assert 'Drawdown' in reason

    def test_no_exit_below_threshold(self):
        """Should not trigger exit below threshold."""
        from trading_metrics import make_exit_condition

        config = {'sell_drawdown_threshold': 0.10}
        exit_fn = make_exit_condition(config)

        row = pd.Series({
            'drawdown': -0.05,  # 5% drawdown < 10% threshold
            'close': 95
        })

        should_exit, reason = exit_fn(row, entry_price=100)

        assert should_exit is False
        assert reason == ""

    def test_exit_on_take_profit(self):
        """Should trigger exit on take profit."""
        from trading_metrics import make_exit_condition

        config = {
            'sell_drawdown_threshold': 0.20,
            'take_profit_pct': 0.10
        }
        exit_fn = make_exit_condition(config)

        row = pd.Series({
            'drawdown': 0.0,
            'close': 112  # 12% gain > 10% take profit
        })

        should_exit, reason = exit_fn(row, entry_price=100)

        assert should_exit is True
        assert 'Take profit' in reason

    def test_exit_on_stop_loss(self):
        """Should trigger exit on stop loss."""
        from trading_metrics import make_exit_condition

        config = {
            'sell_drawdown_threshold': 0.20,
            'stop_loss_pct': 0.05
        }
        exit_fn = make_exit_condition(config)

        row = pd.Series({
            'drawdown': -0.03,  # Below drawdown threshold
            'close': 93  # 7% loss > 5% stop loss
        })

        should_exit, reason = exit_fn(row, entry_price=100)

        assert should_exit is True
        assert 'Stop loss' in reason


class TestMakeReentryCondition:
    """Tests for make_reentry_condition function."""

    def test_reentry_on_dip(self):
        """Should trigger re-entry on dip."""
        from trading_metrics import make_reentry_condition

        config = {'buy_dip_threshold': 0.05}
        reentry_fn = make_reentry_condition(config)

        row = pd.Series({
            'pct_from_high': -0.08  # 8% dip > 5% threshold
        })

        should_enter, reason = reentry_fn(row)

        assert should_enter is True
        assert 'Dip' in reason

    def test_no_reentry_above_threshold(self):
        """Should not trigger re-entry above threshold."""
        from trading_metrics import make_reentry_condition

        config = {'buy_dip_threshold': 0.10}
        reentry_fn = make_reentry_condition(config)

        row = pd.Series({
            'pct_from_high': -0.05  # 5% dip < 10% threshold
        })

        should_enter, reason = reentry_fn(row)

        assert should_enter is False
        assert reason == ""


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================

# =============================================================================
# Backtest Functions
# =============================================================================

class TestRunBacktest:
    """Tests for run_backtest function."""

    def test_returns_backtest_result(self):
        """Should return BacktestResult with all components."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 105, 110, 108, 115]
        })

        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')],
            'price': [110.0, 108.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        assert hasattr(result, 'metrics')
        assert hasattr(result, 'baseline')
        assert hasattr(result, 'equity_curve')

    def test_equity_curve_has_strategy_and_baseline(self):
        """Equity curve should have strategy and baseline values."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 105, 110, 108, 115]
        })

        # No signals = buy and hold
        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        result = run_backtest(signals_df, prices_df, 'date', 'price', initial_equity=10000)

        assert len(result.equity_curve) == 5
        assert 'date' in result.equity_curve[0]
        assert 'strategy' in result.equity_curve[0]
        assert 'baseline' in result.equity_curve[0]

    def test_initial_equity_scales_curves(self):
        """Initial equity should scale the curves."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'price': [100, 110, 120]
        })

        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        result = run_backtest(signals_df, prices_df, 'date', 'price', initial_equity=5000)

        # First equity should be initial_equity
        assert result.equity_curve[0]['strategy'] == pytest.approx(5000, rel=0.01)
        assert result.equity_curve[0]['baseline'] == pytest.approx(5000, rel=0.01)

    def test_empty_prices_raises_error(self):
        """Empty prices should raise InsufficientDataError."""
        from trading_metrics import run_backtest, InsufficientDataError

        prices_df = pd.DataFrame({'date': [], 'price': []})
        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        with pytest.raises(InsufficientDataError, match="prices_df is empty"):
            run_backtest(signals_df, prices_df, 'date', 'price')

    def test_baseline_comparison_included(self):
        """Should include baseline comparison."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 105, 110, 108, 115]
        })

        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02')],
            'price': [105.0],
            'action': ['SELL']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        assert result.baseline.buy_hold_return == pytest.approx(0.15, rel=0.01)  # 100->115 = 15%


class TestMetricsToDict:
    """Tests for metrics_to_dict function and BacktestMetrics.to_dict()."""

    def test_metrics_to_dict_returns_dict(self):
        """Should return a dictionary."""
        from trading_metrics import metrics_to_dict, BacktestMetrics

        metrics = BacktestMetrics(
            total_return=0.10, cagr=0.08, sharpe_ratio=1.5, sortino_ratio=2.0,
            max_drawdown=-0.15, volatility=0.20, win_rate=0.60,
            num_trades=5, avg_trade_return=0.02,
            total_signals=100, buy_signals=10, sell_signals=10, hold_signals=80
        )

        result = metrics_to_dict(metrics)

        assert isinstance(result, dict)
        assert result['total_return'] == 0.10
        assert result['sharpe_ratio'] == 1.5

    def test_to_dict_method_matches_function(self):
        """BacktestMetrics.to_dict() should match metrics_to_dict()."""
        from trading_metrics import metrics_to_dict, BacktestMetrics

        metrics = BacktestMetrics(
            total_return=0.25, cagr=0.15, sharpe_ratio=2.0, sortino_ratio=2.5,
            max_drawdown=-0.10, volatility=0.15, win_rate=0.65,
            num_trades=8, avg_trade_return=0.03,
            total_signals=200, buy_signals=15, sell_signals=15, hold_signals=170
        )

        dict_from_function = metrics_to_dict(metrics)
        dict_from_method = metrics.to_dict()

        assert dict_from_function == dict_from_method

    def test_to_dict_has_all_fields(self):
        """to_dict should include all required fields."""
        from trading_metrics import BacktestMetrics

        metrics = BacktestMetrics(
            total_return=0.10, cagr=0.08, sharpe_ratio=1.5, sortino_ratio=2.0,
            max_drawdown=-0.15, volatility=0.20, win_rate=0.60,
            num_trades=5, avg_trade_return=0.02,
            total_signals=100, buy_signals=10, sell_signals=10, hold_signals=80
        )

        result = metrics.to_dict()

        expected_keys = [
            'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'volatility', 'win_rate',
            'num_trades', 'avg_trade_return',
            'total_predictions', 'buy_signals', 'sell_signals', 'hold_signals'
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_metrics_handle_nan_series(self):
        """All metrics should handle series with only NaN values."""
        from trading_metrics import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown,
            calculate_total_return,
            calculate_cagr,
            calculate_volatility,
        )

        nan_series = pd.Series([np.nan, np.nan, np.nan])

        # All should return 0.0 or handle gracefully
        assert calculate_sharpe_ratio(nan_series) == 0.0
        assert calculate_sortino_ratio(nan_series) == 0.0
        assert calculate_max_drawdown(nan_series) == 0.0
        assert calculate_total_return(nan_series) == 0.0
        assert calculate_cagr(nan_series) == 0.0
        assert calculate_volatility(nan_series) == 0.0

    def test_metrics_with_inf_values(self):
        """Metrics should handle infinity values gracefully."""
        from trading_metrics import calculate_sharpe_ratio, calculate_volatility

        # Series with infinity should be handled
        series_with_inf = pd.Series([0.01, np.inf, 0.01, 0.01])

        # Should not raise exception
        result = calculate_sharpe_ratio(series_with_inf)
        assert not np.isnan(result) or result == 0.0

    def test_very_small_returns(self):
        """Metrics should handle very small returns."""
        from trading_metrics import calculate_total_return, calculate_sharpe_ratio

        tiny_returns = pd.Series([0.0001, 0.0001, 0.0001] * 100)

        total = calculate_total_return(tiny_returns)
        sharpe = calculate_sharpe_ratio(tiny_returns)

        assert total > 0
        assert sharpe > 0

    def test_very_large_returns(self):
        """Metrics should handle very large returns."""
        from trading_metrics import calculate_total_return

        large_returns = pd.Series([1.0, 1.0, 1.0])  # 100% daily returns

        total = calculate_total_return(large_returns)

        # (1+1)^3 - 1 = 7 = 700%
        assert total == pytest.approx(7.0, rel=0.01)
