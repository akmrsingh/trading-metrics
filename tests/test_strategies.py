"""
Tests for Strategy Signal Generators
=====================================

Tests for generate_signals() and backtest_strategy() functions.

Run with: pytest tests/test_strategies.py -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestGenerateSignals:
    """Tests for generate_signals function."""

    @pytest.fixture
    def price_data_with_dip(self):
        """Price data with a clear dip pattern."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        # Price rises to 110, drops to 95 (dip), recovers to 120
        prices = []
        for i in range(60):
            if i < 20:
                prices.append(100 + i * 0.5)  # Rise to 110
            elif i < 35:
                prices.append(110 - (i - 20) * 1.0)  # Drop to 95
            else:
                prices.append(95 + (i - 35) * 1.0)  # Recover to 120

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices,
            'volume': [1000000] * 60
        })

    @pytest.fixture
    def price_data_steady_rise(self):
        """Price data with steady rise (no signals expected)."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        prices = [100 + i * 0.5 for i in range(60)]  # Steady rise

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices,
            'volume': [1000000] * 60
        })

    def test_stoploss_generates_sell_on_drawdown(self, price_data_with_dip):
        """Test that stoploss strategy generates SELL on drawdown."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.08
            },
            prices_df=price_data_with_dip,
            symbol="QQQ"
        )

        # Should have at least one SELL (when drawdown hits threshold)
        sells = signals[signals['action'] == 'SELL']
        assert len(sells) >= 1, "Should generate SELL on drawdown"

        # SELL reason should mention drawdown
        assert 'Drawdown' in sells.iloc[0]['reason']

    def test_stoploss_generates_buy_on_dip(self, price_data_with_dip):
        """Test that stoploss strategy generates BUY on dip."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.08
            },
            prices_df=price_data_with_dip,
            symbol="QQQ"
        )

        # Should have BUY after SELL (when dip condition met)
        buys = signals[signals['action'] == 'BUY']
        if len(signals[signals['action'] == 'SELL']) > 0:
            assert len(buys) >= 1, "Should generate BUY after SELL when dip occurs"

    def test_no_signals_on_steady_rise(self, price_data_steady_rise):
        """Test that no signals generated during steady rise."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.08
            },
            prices_df=price_data_steady_rise,
            symbol="QQQ"
        )

        # No drawdown = no SELL, already invested = no BUY
        assert len(signals) == 0, "Should not generate signals during steady rise"

    def test_signals_include_required_columns(self, price_data_with_dip):
        """Test that signals DataFrame has required columns."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.10
            },
            prices_df=price_data_with_dip,
            symbol="QQQ"
        )

        required_columns = ['date', 'action', 'price', 'reason', 'symbol']
        for col in required_columns:
            assert col in signals.columns, f"Missing column: {col}"

    def test_symbol_in_output(self, price_data_with_dip):
        """Test that symbol is included in signal output."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.10},
            prices_df=price_data_with_dip,
            symbol="TQQQ"
        )

        if len(signals) > 0:
            assert all(signals['symbol'] == 'TQQQ'), "Symbol should be in output"

    def test_unknown_model_type_raises_error(self, price_data_with_dip):
        """Test that unknown model type raises ValueError."""
        from trading_metrics import generate_signals

        with pytest.raises(ValueError, match="Unknown model type"):
            generate_signals(
                model_type="unknown_strategy",
                config={},
                prices_df=price_data_with_dip,
                symbol="QQQ"
            )

    def test_take_profit_signal(self):
        """Test take profit exit condition."""
        from trading_metrics import generate_signals

        # Start at 100, rise to 120 (20% gain)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices = [100 + i * 0.8 for i in range(30)]

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.20,
                "take_profit_pct": 0.10  # Take profit at 10%
            },
            prices_df=df,
            symbol="QQQ"
        )

        # Should have SELL for take profit
        sells = signals[signals['action'] == 'SELL']
        if len(sells) > 0:
            assert 'Take profit' in sells.iloc[0]['reason']

    def test_stop_loss_signal(self):
        """Test stop loss exit condition."""
        from trading_metrics import generate_signals

        # Start at 100, drop to 85 (15% loss)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices = [100 - i * 0.5 for i in range(30)]

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

        signals = generate_signals(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.05,
                "sell_drawdown_threshold": 0.20,
                "stop_loss_pct": 0.05  # Stop loss at 5%
            },
            prices_df=df,
            symbol="QQQ"
        )

        # Should have SELL for stop loss
        sells = signals[signals['action'] == 'SELL']
        assert len(sells) >= 1


class TestMomentumSignals:
    """Tests for momentum/MA crossover strategy."""

    @pytest.fixture
    def trending_data(self):
        """Data with clear trend changes for MA crossover."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        # Rising trend, then falling, then rising again
        prices = []
        for i in range(100):
            if i < 40:
                prices.append(100 + i * 1.0)  # Rise
            elif i < 70:
                prices.append(140 - (i - 40) * 1.5)  # Fall
            else:
                prices.append(95 + (i - 70) * 1.2)  # Rise again

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

    def test_momentum_generates_crossover_signals(self, trending_data):
        """Test that momentum strategy generates signals on MA crossovers."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="momentum",
            config={
                "fast_period": 5,
                "slow_period": 20
            },
            prices_df=trending_data,
            symbol="SPY"
        )

        # Should have at least one SELL (bearish crossover)
        sells = signals[signals['action'] == 'SELL']
        assert len(sells) >= 1, "Should generate SELL on bearish crossover"

        # Reason should mention crossover
        assert 'crossover' in sells.iloc[0]['reason'].lower()


class TestBacktestStrategy:
    """Tests for backtest_strategy convenience function."""

    @pytest.fixture
    def price_data(self):
        """Price data for backtesting."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        # Volatile data with dips and recoveries
        prices = []
        for i in range(60):
            base = 100 + i * 0.3
            noise = 5 * np.sin(i * 0.3)
            prices.append(base + noise)

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices]
        })

    def test_backtest_returns_result(self, price_data):
        """Test that backtest_strategy returns BacktestResult."""
        from trading_metrics import backtest_strategy

        result = backtest_strategy(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.03,
                "sell_drawdown_threshold": 0.05
            },
            prices_df=price_data,
            symbol="QQQ"
        )

        # Should return BacktestResult
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'baseline')
        assert result.metrics is not None

    def test_backtest_metrics_reasonable(self, price_data):
        """Test that backtest metrics are within reasonable bounds."""
        from trading_metrics import backtest_strategy

        result = backtest_strategy(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.03,
                "sell_drawdown_threshold": 0.05
            },
            prices_df=price_data,
            symbol="QQQ"
        )

        # Metrics should be reasonable
        assert -1.0 <= result.metrics.total_return <= 10.0
        assert result.metrics.max_drawdown <= 0
        assert -10 <= result.metrics.sharpe_ratio <= 10

    def test_backtest_includes_baseline(self, price_data):
        """Test that backtest includes baseline comparison."""
        from trading_metrics import backtest_strategy

        result = backtest_strategy(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.03,
                "sell_drawdown_threshold": 0.05
            },
            prices_df=price_data,
            symbol="QQQ"
        )

        # Should have baseline
        assert result.baseline is not None
        assert hasattr(result.baseline, 'buy_hold_return')
        assert hasattr(result.baseline, 'outperformance')

    def test_backtest_with_date_range(self, price_data):
        """Test backtest with specific date range."""
        from trading_metrics import backtest_strategy

        result = backtest_strategy(
            model_type="stoploss",
            config={
                "buy_dip_threshold": 0.03,
                "sell_drawdown_threshold": 0.05
            },
            prices_df=price_data,
            symbol="QQQ",
            start_date="2024-01-15",
            end_date="2024-02-28"
        )

        assert result.metrics is not None


class TestIntegration:
    """Integration tests for full workflow."""

    def test_unified_workflow(self):
        """Test the complete unified workflow: model+config → signals → metrics."""
        from trading_metrics import generate_signals, run_backtest

        # 1. Create price data
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        prices = [100 + i * 0.3 + 3 * np.sin(i * 0.2) for i in range(60)]

        prices_df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices]
        })

        # 2. MODEL + CONFIG → STRATEGY
        model_type = "stoploss"
        config = {
            "buy_dip_threshold": 0.03,
            "sell_drawdown_threshold": 0.05
        }
        symbol = "QQQ"

        # 3. + SYMBOL + DATE_RANGE → SPARSE PREDICTIONS
        signals_df = generate_signals(
            model_type=model_type,
            config=config,
            prices_df=prices_df,
            symbol=symbol
        )

        # Verify signals format
        assert 'date' in signals_df.columns
        assert 'action' in signals_df.columns
        assert 'price' in signals_df.columns
        assert 'symbol' in signals_df.columns

        # 4. + PRICES → METRICS (run_backtest takes signals + prices)
        # Create price df for backtest (date + close column)
        backtest_prices = prices_df[['date', 'close']].copy()

        result = run_backtest(
            signals_df=signals_df,
            prices_df=backtest_prices,
            date_col='date',
            price_col='close',
            signal_col='action'
        )

        # Verify metrics
        assert result.metrics is not None
        assert result.baseline is not None
        assert result.baseline.buy_hold_return is not None

    def test_same_strategy_different_symbols(self):
        """Test same strategy on different symbols produces different results."""
        from trading_metrics import generate_signals

        # Same config
        config = {"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08}

        # Different price data (simulating different symbols)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")

        qqq_prices = [100 + i * 0.3 for i in range(60)]
        tqqq_prices = [50 + i * 0.9 for i in range(60)]  # 3x leverage (more volatile)

        qqq_df = pd.DataFrame({
            'date': dates, 'close': qqq_prices,
            'high': [p * 1.01 for p in qqq_prices], 'low': [p * 0.99 for p in qqq_prices]
        })

        tqqq_df = pd.DataFrame({
            'date': dates, 'close': tqqq_prices,
            'high': [p * 1.03 for p in tqqq_prices], 'low': [p * 0.97 for p in tqqq_prices]
        })

        qqq_signals = generate_signals("stoploss", config, qqq_df, "QQQ")
        tqqq_signals = generate_signals("stoploss", config, tqqq_df, "TQQQ")

        # Symbols should be in output
        if len(qqq_signals) > 0:
            assert all(qqq_signals['symbol'] == 'QQQ')
        if len(tqqq_signals) > 0:
            assert all(tqqq_signals['symbol'] == 'TQQQ')


class TestHoldSignals:
    """Tests for include_hold parameter in generate_signals."""

    @pytest.fixture
    def price_data(self):
        """Price data with some volatility."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        # Steady prices with small fluctuations (no triggers)
        prices = [100 + i * 0.1 for i in range(30)]

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

    @pytest.fixture
    def price_data_with_dip(self):
        """Price data with a dip pattern."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        prices = []
        for i in range(60):
            if i < 20:
                prices.append(100 + i * 0.5)
            elif i < 35:
                prices.append(110 - (i - 20) * 1.0)
            else:
                prices.append(95 + (i - 35) * 1.0)

        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

    def test_include_hold_generates_one_signal_per_day(self, price_data):
        """With include_hold=True, should have one signal per trading day."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
            prices_df=price_data,
            symbol="QQQ",
            include_hold=True
        )

        # Should have exactly one signal per trading day
        assert len(signals) == len(price_data), f"Expected {len(price_data)} signals, got {len(signals)}"

    def test_include_hold_false_is_sparse(self, price_data):
        """With include_hold=False (default), should be sparse (no HOLD)."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
            prices_df=price_data,
            symbol="QQQ",
            include_hold=False
        )

        # Sparse = fewer signals than days
        assert len(signals) < len(price_data)

        # No HOLD signals in sparse mode
        if len(signals) > 0:
            assert 'HOLD' not in signals['action'].values

    def test_hold_is_real_model_decision(self, price_data_with_dip):
        """HOLD signals should have meaningful reasons (not synthetic)."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
            prices_df=price_data_with_dip,
            symbol="QQQ",
            include_hold=True
        )

        hold_signals = signals[signals['action'] == 'HOLD']
        assert len(hold_signals) > 0, "Should have some HOLD signals"

        # HOLD reasons should be meaningful
        for _, row in hold_signals.iterrows():
            reason = row['reason']
            assert reason is not None
            assert len(reason) > 0
            # Reasons should indicate position state
            assert any(keyword in reason.lower() for keyword in ['hold', 'wait', 'position'])

    def test_buy_sell_hold_only_actions(self, price_data_with_dip):
        """All signals should be BUY, SELL, or HOLD."""
        from trading_metrics import generate_signals

        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
            prices_df=price_data_with_dip,
            symbol="QQQ",
            include_hold=True
        )

        valid_actions = {'BUY', 'SELL', 'HOLD'}
        actual_actions = set(signals['action'].unique())

        assert actual_actions.issubset(valid_actions), f"Invalid actions: {actual_actions - valid_actions}"

    def test_momentum_include_hold(self):
        """Momentum strategy should also support include_hold."""
        from trading_metrics import generate_signals

        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        prices = [100 + i * 0.5 + 5 * np.sin(i * 0.3) for i in range(50)]

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices]
        })

        signals = generate_signals(
            model_type="momentum",
            config={"fast_period": 5, "slow_period": 20},
            prices_df=df,
            symbol="SPY",
            include_hold=True
        )

        # Should have one signal per day
        assert len(signals) == len(df)

        # Should have HOLD signals
        assert 'HOLD' in signals['action'].values


class TestGapValidation:
    """Tests for validate_completeness parameter in run_backtest."""

    @pytest.fixture
    def price_data(self):
        """Daily price data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = [100 + i for i in range(10)]

        return pd.DataFrame({
            'date': dates,
            'close': prices
        })

    @pytest.fixture
    def complete_signals(self, price_data):
        """Complete signals (one per day)."""
        return pd.DataFrame({
            'date': price_data['date'],
            'action': ['HOLD'] * 10,
            'price': price_data['close']
        })

    @pytest.fixture
    def incomplete_signals(self, price_data):
        """Incomplete signals (missing some days)."""
        # Only include every other day
        return pd.DataFrame({
            'date': price_data['date'].iloc[::2],  # 0, 2, 4, 6, 8
            'action': ['HOLD'] * 5,
            'price': price_data['close'].iloc[::2]
        })

    def test_validate_completeness_passes_for_complete(self, complete_signals, price_data):
        """Should pass when all trading days have signals."""
        from trading_metrics import run_backtest

        # Should not raise
        result = run_backtest(
            signals_df=complete_signals,
            prices_df=price_data,
            date_col='date',
            price_col='close',
            validate_completeness=True
        )

        assert result is not None

    def test_validate_completeness_fails_for_gaps(self, incomplete_signals, price_data):
        """Should raise InvalidDataError when trading days are missing."""
        from trading_metrics import run_backtest, InvalidDataError

        with pytest.raises(InvalidDataError, match="Missing predictions"):
            run_backtest(
                signals_df=incomplete_signals,
                prices_df=price_data,
                date_col='date',
                price_col='close',
                validate_completeness=True
            )

    def test_validate_completeness_off_allows_gaps(self, incomplete_signals, price_data):
        """With validate_completeness=False, gaps should be allowed."""
        from trading_metrics import run_backtest

        # Should not raise
        result = run_backtest(
            signals_df=incomplete_signals,
            prices_df=price_data,
            date_col='date',
            price_col='close',
            validate_completeness=False  # Default
        )

        assert result is not None

    def test_generate_signals_with_hold_for_gap_detection(self, price_data):
        """Full workflow: generate with HOLD, backtest with validation."""
        from trading_metrics import generate_signals, run_backtest

        # Add required columns for generate_signals
        full_data = price_data.copy()
        full_data['high'] = full_data['close'] * 1.01
        full_data['low'] = full_data['close'] * 0.99

        # Generate with HOLD signals
        signals = generate_signals(
            model_type="stoploss",
            config={"buy_dip_threshold": 0.05, "sell_drawdown_threshold": 0.08},
            prices_df=full_data,
            symbol="QQQ",
            include_hold=True
        )

        # Backtest with validation should pass
        result = run_backtest(
            signals_df=signals,
            prices_df=price_data,
            date_col='date',
            price_col='close',
            validate_completeness=True
        )

        assert result is not None
        assert len(signals) == len(price_data)
