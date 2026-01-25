# Trading Metrics

Shared performance metrics library for trading projects.

## Installation

```bash
# From local path
pip install -e /path/to/trading-metrics

# From GitHub (after pushing)
pip install git+https://github.com/YOUR_USERNAME/trading-metrics.git
```

## Usage

```python
from trading_metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    run_backtest
)

import pandas as pd

# Calculate individual metrics
returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(returns)
total_ret = calculate_total_return(returns)

# Or run full backtest
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'price': [100, 102, 101],
    'action': ['BUY', 'HOLD', 'SELL']
})
metrics = run_backtest(df)
print(f"Total Return: {metrics.total_return:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

## Available Functions

### Core Metrics
| Function | Description |
|----------|-------------|
| `calculate_sharpe_ratio(returns)` | Annualized Sharpe ratio |
| `calculate_sortino_ratio(returns)` | Annualized Sortino ratio |
| `calculate_max_drawdown(returns)` | Maximum drawdown |
| `calculate_total_return(returns)` | Total cumulative return |
| `calculate_cagr(returns)` | Compound Annual Growth Rate |
| `calculate_volatility(returns)` | Annualized volatility |

### Win Rate Variants
| Function | Description |
|----------|-------------|
| `calculate_trade_win_rate(trades)` | % of profitable trades |
| `calculate_daily_win_rate(returns)` | % of profitable days |
| `calculate_monthly_win_rate(returns)` | % of profitable months |

### High-Level
| Function | Description |
|----------|-------------|
| `run_backtest(df)` | Run complete backtest from signals |
| `simulate_trades(df)` | Simulate BUY/SELL trades |

## Formula Reference

### Sharpe Ratio
```
Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
```

### Max Drawdown
```
cumulative = cumprod(1 + returns)
running_max = expanding_max(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = min(drawdown)
```

### Total Return
```
total_return = prod(1 + returns) - 1
```

## Dependencies

- pandas
- numpy
- quantstats (for validated calculations)
