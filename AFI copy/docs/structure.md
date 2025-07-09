# AFI Trading System Directory Structure

This document outlines the directory and file structure of the AFI trading system.

```
AFI/
├── README.md
├── agents/
│   ├── __init__.py
│   └── orchestrator.py
├── api/
│   ├── __init__.py
│   └── server.py
├── config.yaml
├── core/
│   ├── __init__.py
│   ├── exchange_config.py
│   └── market_analyzer.py
├── docs/
│   ├── debug.py
│   ├── documentation.md
│   ├── roadmap.md
│   └── structure.md
├── execution/
│   ├── __init__.py
│   └── order_executor.py
├── logs/
├── main.py
├── requirements.txt
├── run_paper_trade.py
└── strategies/
    ├── __init__.py
    ├── base_strat.py
    ├── bear.py
    ├── bull.py
    ├── ranging.py
    └── volatile.py
```

## File Descriptions

- `AFI/README.md`: Main project README with overview, installation, and usage instructions.
- `AFI/agents/orchestrator.py`: Manages trading strategies based on market conditions.
- `AFI/api/server.py`: FastAPI server for API endpoints and WebSocket communication.
- `AFI/config.yaml`: Configuration file for the entire system.
- `AFI/core/exchange_config.py`: Manages exchange API keys and settings.
- `AFI/core/market_analyzer.py`: Analyzes market data to identify regimes and trends.
- `AFI/docs/`: Contains project documentation.
- `AFI/execution/order_executor.py`: Handles the execution of trades.
- `AFI/logs/`: Directory for log files.
- `AFI/main.py`: The main entry point for the application.
- `AFI/requirements.txt`: Lists the Python dependencies for the project.
- `AFI/run_paper_trade.py`: A script to run the system in paper trading mode.
- `AFI/strategies/`: Contains the different trading strategies.
- `AFI/strategies/base_strat.py`: The base class for all trading strategies.
- `AFI/strategies/bear.py`: Strategy for bearish market conditions.
- `AFI/strategies/bull.py`: Strategy for bullish market conditions.
- `AFI/strategies/ranging.py`: Strategy for ranging market conditions.
- `AFI/strategies/volatile.py`: Strategy for volatile market conditions.
