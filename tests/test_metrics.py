"""
Unit Tests for Trading Metrics Library
======================================

Run with: pytest tests/test_metrics.py -v
"""
import pytest
import pandas as pd
import numpy as np


class TestBaselineComparison:
    """Tests for baseline comparison functions."""

    def test_calculate_buy_hold_return_basic(self):
        """Buy-and-hold return should be (last - first) / first."""
        from trading_metrics import calculate_buy_hold_return

        prices = pd.Series([100, 105, 110, 108, 120])
        result = calculate_buy_hold_return(prices)

        expected = (120 - 100) / 100  # 0.20 = 20%
        assert abs(result - expected) < 0.001

    def test_calculate_buy_hold_return_negative(self):
        """Should handle negative returns correctly."""
        from trading_metrics import calculate_buy_hold_return

        prices = pd.Series([100, 95, 90, 85, 80])
        result = calculate_buy_hold_return(prices)

        expected = (80 - 100) / 100  # -0.20 = -20%
        assert abs(result - expected) < 0.001

    def test_calculate_buy_hold_return_empty(self):
        """Should return 0 for empty series."""
        from trading_metrics import calculate_buy_hold_return

        result = calculate_buy_hold_return(pd.Series([]))
        assert result == 0.0

    def test_calculate_buy_hold_return_single(self):
        """Should return 0 for single value."""
        from trading_metrics import calculate_buy_hold_return

        result = calculate_buy_hold_return(pd.Series([100]))
        assert result == 0.0

    def test_calculate_buy_hold_return_with_start_end_prices(self):
        """Buy-and-hold with start/end prices should match formula."""
        from trading_metrics import calculate_buy_hold_return

        # Test positive return
        result = calculate_buy_hold_return(start_price=100, end_price=120)
        assert result == pytest.approx(0.20, rel=0.001)  # 20%

        # Test negative return
        result = calculate_buy_hold_return(start_price=100, end_price=80)
        assert result == pytest.approx(-0.20, rel=0.001)  # -20%

        # Test zero return
        result = calculate_buy_hold_return(start_price=100, end_price=100)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_calculate_buy_hold_return_methods_equivalent(self):
        """Both methods should give same result for equivalent inputs."""
        from trading_metrics import calculate_buy_hold_return

        prices = pd.Series([100, 105, 110, 108, 120])

        # Method 1: price series
        result_series = calculate_buy_hold_return(prices)

        # Method 2: start/end prices
        result_prices = calculate_buy_hold_return(start_price=100, end_price=120)

        assert result_series == pytest.approx(result_prices, rel=0.001)

    def test_calculate_buy_hold_return_zero_start_price(self):
        """Should handle zero start price safely."""
        from trading_metrics import calculate_buy_hold_return

        result = calculate_buy_hold_return(start_price=0, end_price=100)
        assert result == 0.0  # Should not divide by zero

    def test_calculate_buy_hold_return_realistic_tqqq(self):
        """Test with realistic TQQQ-like numbers."""
        from trading_metrics import calculate_buy_hold_return

        # TQQQ going from $50 to $45 = -10%
        result = calculate_buy_hold_return(start_price=50.0, end_price=45.0)
        assert result == pytest.approx(-0.10, rel=0.001)

        # TQQQ going from $50 to $75 = +50%
        result = calculate_buy_hold_return(start_price=50.0, end_price=75.0)
        assert result == pytest.approx(0.50, rel=0.001)


class TestTradeAnalysis:
    """Tests for trade cycle analysis."""

    def test_analyze_exit_reentry_avoided_drop(self):
        """Should detect avoided drop when market falls."""
        from trading_metrics import analyze_exit_reentry

        result = analyze_exit_reentry(
            sell_date="2024-01-15",
            sell_price=100.0,
            sell_reason="Drawdown",
            buy_date="2024-01-25",
            buy_price=90.0,  # Market dropped 10%
            buy_reason="Dip buy"
        )

        assert result.price_change_while_out == pytest.approx(-0.10, rel=0.01)
        assert result.benefit == pytest.approx(0.10, rel=0.01)  # We avoided 10% drop
        assert "Avoided" in result.analysis
        assert "10" in result.analysis

    def test_analyze_exit_reentry_missed_gain(self):
        """Should detect missed gain when market rises."""
        from trading_metrics import analyze_exit_reentry

        result = analyze_exit_reentry(
            sell_date="2024-01-15",
            sell_price=100.0,
            sell_reason="VIX spike",
            buy_date="2024-01-25",
            buy_price=115.0,  # Market rose 15%
            buy_reason="Dip buy"
        )

        assert result.price_change_while_out == pytest.approx(0.15, rel=0.01)
        assert result.benefit == pytest.approx(-0.15, rel=0.01)  # We missed 15% gain
        assert "Missed" in result.analysis
        assert "15" in result.analysis

    def test_analyze_exit_reentry_neutral(self):
        """Should detect neutral when price unchanged."""
        from trading_metrics import analyze_exit_reentry

        result = analyze_exit_reentry(
            sell_date="2024-01-15",
            sell_price=100.0,
            sell_reason="Risk off",
            buy_date="2024-01-25",
            buy_price=100.0,  # Price unchanged
            buy_reason="Re-entry"
        )

        assert result.price_change_while_out == pytest.approx(0.0, abs=0.01)
        assert "Neutral" in result.analysis


class TestCoreMetrics:
    """Tests for core metric calculations."""

    def test_sharpe_ratio_positive(self):
        """Sharpe ratio should be positive for good returns."""
        from trading_metrics import calculate_sharpe_ratio

        # Consistent positive returns
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01] * 50)
        result = calculate_sharpe_ratio(returns)

        assert result > 0

    def test_sharpe_ratio_negative(self):
        """Sharpe ratio should be negative for bad returns."""
        from trading_metrics import calculate_sharpe_ratio

        # Consistent negative returns
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01] * 50)
        result = calculate_sharpe_ratio(returns)

        assert result < 0

    def test_max_drawdown_always_negative_or_zero(self):
        """Max drawdown should always be <= 0."""
        from trading_metrics import calculate_max_drawdown

        # Even with positive returns, max_dd should be <= 0
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01] * 50)
        result = calculate_max_drawdown(returns)

        assert result <= 0

    def test_max_drawdown_detects_dip(self):
        """Max drawdown should detect significant dips."""
        from trading_metrics import calculate_max_drawdown

        # Series with a 20% drawdown
        returns = pd.Series([0.05, 0.05, -0.15, -0.10, 0.05, 0.05])
        result = calculate_max_drawdown(returns)

        # Should detect at least 20% drawdown
        assert result < -0.15

    def test_total_return_calculation(self):
        """Total return should compound correctly."""
        from trading_metrics import calculate_total_return

        # 10% gain, then 10% loss: (1.1 * 0.9) - 1 = -0.01
        returns = pd.Series([0.10, -0.10])
        result = calculate_total_return(returns)

        expected = (1.10 * 0.90) - 1
        assert abs(result - expected) < 0.001

    def test_volatility_positive(self):
        """Volatility should always be positive."""
        from trading_metrics import calculate_volatility

        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
        result = calculate_volatility(returns)

        assert result > 0


class TestWinRates:
    """Tests for win rate calculations."""

    def test_trade_win_rate(self):
        """Trade win rate should count winning trades."""
        from trading_metrics import calculate_trade_win_rate, Trade

        trades = [
            Trade("2024-01-01", "2024-01-10", 100, 110, 0.10),  # Win
            Trade("2024-01-11", "2024-01-20", 110, 100, -0.09),  # Loss
            Trade("2024-01-21", "2024-01-30", 100, 105, 0.05),  # Win
        ]

        result = calculate_trade_win_rate(trades)
        assert result == pytest.approx(2/3, rel=0.01)  # 66.7%


class TestDataClasses:
    """Tests for dataclass serialization."""

    def test_trade_analysis_dataclass(self):
        """TradeAnalysis should have all expected fields."""
        from trading_metrics import TradeAnalysis

        ta = TradeAnalysis(
            sell_date="2024-01-15",
            sell_price=100.0,
            sell_reason="Test",
            buy_date="2024-01-20",
            buy_price=95.0,
            buy_reason="Test",
            price_change_while_out=-0.05,
            benefit=0.05,
            analysis="Avoided 5% drop"
        )

        assert ta.sell_date == "2024-01-15"
        assert ta.benefit == 0.05
        assert "Avoided" in ta.analysis

    def test_baseline_comparison_dataclass(self):
        """BaselineComparison should have all expected fields."""
        from trading_metrics import BaselineComparison

        bc = BaselineComparison(
            strategy_return=0.25,
            buy_hold_return=0.20,
            outperformance=0.05,
            outperformance_pct=4.17  # (1.25/1.20 - 1) * 100
        )

        assert bc.strategy_return == 0.25
        assert bc.outperformance == 0.05
        assert bc.outperformance_pct == 4.17


class TestRunBacktest:
    """Tests for run_backtest function - the main backtest entry point."""

    def test_run_backtest_no_signals(self):
        """No signals = buy and hold, should match B&H return."""
        from trading_metrics import run_backtest, calculate_buy_hold_return

        # Price data: 100 to 120 (20% return)
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 105, 110, 115, 120]
        })

        # Empty signals = stayed invested
        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # Strategy return should equal B&H since no trades
        assert result.baseline.buy_hold_return == pytest.approx(0.20, rel=0.01)
        assert result.metrics.total_return == pytest.approx(0.20, rel=0.01)

    def test_run_backtest_sell_at_top_buy_at_bottom(self):
        """Perfect timing: sell before drop, buy at bottom."""
        from trading_metrics import run_backtest

        # Price: 100 -> 110 -> 90 -> 100
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'price': [100.0, 110.0, 90.0, 100.0]
        })

        # Sell at 110, buy at 90
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')],
            'price': [110.0, 90.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # Started invested at 100, grew to 110, sold
        # Bought back at 90, ended at 100
        # Return: (110/100) * (100/90) - 1 = 1.1 * 1.111 - 1 = 0.222 (22.2%)
        assert result.metrics.total_return > 0.20  # Beat B&H (0%)
        assert result.metrics.num_trades >= 1

    def test_run_backtest_bad_timing(self):
        """Bad timing: sell at bottom, buy at top."""
        from trading_metrics import run_backtest

        # Price: 100 -> 90 -> 110 -> 100
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'price': [100.0, 90.0, 110.0, 100.0]
        })

        # Sell at 90 (bottom), buy at 110 (top)
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')],
            'price': [90.0, 110.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # Started at 100, dropped to 90, sold (realized -10%)
        # Bought at 110, dropped to 100 (another loss)
        # Return should be negative
        assert result.metrics.total_return < 0

    def test_run_backtest_realistic_scenario(self):
        """Test with realistic trading scenario."""
        from trading_metrics import run_backtest

        # Simulate a market drop and recovery
        # Price: 100 -> 95 -> 85 -> 80 -> 85 -> 95 -> 100
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'price': [100.0, 95.0, 85.0, 80.0, 85.0, 95.0, 100.0]
        })

        # Sell at 95, buy at 80
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-04')],
            'price': [95.0, 80.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # B&H return = 0% (100 -> 100)
        assert result.baseline.buy_hold_return == pytest.approx(0.0, abs=0.01)

        # Strategy should beat B&H
        assert result.metrics.total_return > result.baseline.buy_hold_return

    def test_run_backtest_num_trades(self):
        """Verify trade count is correct."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=6),
            'price': [100, 105, 100, 110, 100, 105]
        })

        # 2 SELL-BUY cycles
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03'),
                     pd.Timestamp('2024-01-04'), pd.Timestamp('2024-01-05')],
            'price': [105.0, 100.0, 110.0, 100.0],
            'action': ['SELL', 'BUY', 'SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # 2 trade cycles
        assert result.metrics.num_trades == 2

    def test_run_backtest_empty_prices(self):
        """Should raise InsufficientDataError for empty prices."""
        from trading_metrics import run_backtest, InsufficientDataError

        prices_df = pd.DataFrame({'date': [], 'price': []})
        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        with pytest.raises(InsufficientDataError, match="prices_df is empty"):
            run_backtest(signals_df, prices_df, 'date', 'price')

    def test_run_backtest_single_price(self):
        """Should raise InsufficientDataError for single price."""
        from trading_metrics import run_backtest, InsufficientDataError

        prices_df = pd.DataFrame({
            'date': ['2024-01-01'],
            'price': [100]
        })
        signals_df = pd.DataFrame({'date': [], 'price': [], 'action': []})

        with pytest.raises(InsufficientDataError, match="need at least 2"):
            run_backtest(signals_df, prices_df, 'date', 'price')

    def test_run_backtest_equity_curve(self):
        """Should return equity curve with daily values."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 105, 110, 108, 115]
        })

        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-03')],
            'price': [110.0],
            'action': ['SELL']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # Should have equity curve with 5 points
        assert len(result.equity_curve) == 5
        assert 'date' in result.equity_curve[0]
        assert 'strategy' in result.equity_curve[0]
        assert 'baseline' in result.equity_curve[0]

    def test_run_backtest_outperformance(self):
        """Test basic SELL then BUY cycle with outperformance."""
        from trading_metrics import run_backtest

        # Price goes: 100 -> 110 -> 90 -> 100
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'price': [100.0, 110.0, 90.0, 100.0]
        })

        # SELL at 110, BUY at 90
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')],
            'price': [110.0, 90.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # B&H: 100 -> 100 = 0%
        assert result.baseline.buy_hold_return == pytest.approx(0.0, abs=0.01)

        # Strategy should outperform
        assert result.metrics.total_return > 0.15
        assert result.baseline.outperformance > 0.15

    def test_run_backtest_hold_only_equals_buy_hold(self):
        """HOLD-only signals (0 trades) should have return exactly equal to B&H.

        This is a critical invariant: if no BUY/SELL signals, strategy = buy-and-hold.
        Bug found: when signal prices differ from raw prices, returns diverged.
        """
        from trading_metrics import run_backtest

        # Raw prices: 100 -> 120 (20% return)
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100.0, 105.0, 110.0, 115.0, 120.0]
        })

        # HOLD-only signals with DIFFERENT prices (simulating prediction vs close)
        # This is the bug scenario: signal prices differ from raw prices
        signals_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [99.0, 104.0, 109.0, 114.0, 119.0],  # Different from close!
            'action': ['HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'close', signal_col='action')

        # With 0 trades, strategy return MUST equal B&H return
        assert result.metrics.num_trades == 0, "Should have 0 trades with HOLD-only signals"
        assert result.baseline.buy_hold_return == pytest.approx(0.20, rel=0.01)

        # THE KEY ASSERTION: strategy return must equal B&H when no trades
        assert result.metrics.total_return == pytest.approx(
            result.baseline.buy_hold_return, rel=0.01
        ), f"With 0 trades, strategy ({result.metrics.total_return:.4f}) must equal B&H ({result.baseline.buy_hold_return:.4f})"

        # Outperformance must be ~0 with no trades
        assert result.baseline.outperformance == pytest.approx(0.0, abs=0.01), \
            f"With 0 trades, outperformance should be 0, got {result.baseline.outperformance:.4f}"


class TestMetricsIntegration:
    """Integration tests verifying metrics work together correctly."""

    def test_strategy_vs_buy_hold_calculation(self):
        """Verify strategy return and B&H are calculated consistently."""
        from trading_metrics import run_backtest, calculate_buy_hold_return

        # Market drops then recovers: B&H = -10%
        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'price': [100, 95, 90, 85, 80, 85, 90]
        })

        # Sell at 90, buy at 80
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-05')],
            'price': [90.0, 80.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')
        bh_return = calculate_buy_hold_return(prices_df['price'])

        # B&H should be -10%
        assert bh_return == pytest.approx(-0.10, rel=0.01)

        # Strategy should beat B&H
        assert result.metrics.total_return > bh_return

    def test_outperformance_calculation(self):
        """Outperformance = strategy_return - buy_hold_return."""
        from trading_metrics import run_backtest

        prices_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'price': [100, 110, 100, 90, 100]
        })

        # Sell at 110, buy at 90
        signals_df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-04')],
            'price': [110.0, 90.0],
            'action': ['SELL', 'BUY']
        })

        result = run_backtest(signals_df, prices_df, 'date', 'price')

        # B&H = 0% (100 -> 100)
        assert result.baseline.buy_hold_return == pytest.approx(0.0, abs=0.01)

        # Strategy should have positive return
        assert result.metrics.total_return > 0
        assert result.baseline.outperformance > 0
