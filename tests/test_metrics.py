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

    def test_compare_to_baseline(self):
        """Compare strategy return to baseline."""
        from trading_metrics import compare_to_baseline

        prices = pd.Series([100, 110, 120])  # B&H = 20%
        strategy_return = 0.25  # Strategy = 25%

        result = compare_to_baseline(strategy_return, prices)

        assert abs(result.buy_hold_return - 0.20) < 0.001
        assert abs(result.strategy_return - 0.25) < 0.001
        assert abs(result.outperformance - 0.05) < 0.001


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

    def test_daily_win_rate(self):
        """Daily win rate should count positive days."""
        from trading_metrics import calculate_daily_win_rate

        returns = pd.Series([0.01, -0.01, 0.02, 0.01, -0.02])  # 3 wins, 2 losses
        result = calculate_daily_win_rate(returns)

        assert result == pytest.approx(3/5, rel=0.01)  # 60%


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
            outperformance=0.05
        )

        assert bc.strategy_return == 0.25
        assert bc.outperformance == 0.05
