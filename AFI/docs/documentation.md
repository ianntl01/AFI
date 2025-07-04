# AFI Trading System Documentation

This document provides a detailed explanation of each component in the AFI trading system.

## System Overview

The AFI trading system is an automated cryptocurrency trading bot that uses a modular architecture to analyze market conditions, manage risk, and execute trades. It is designed to be highly configurable and extensible, allowing for the easy addition of new trading strategies and components.

## Core Components

### 1. Main Entry Point (`main.py`)

- **Purpose**: Initializes and orchestrates all components of the trading system.
- **Functionality**:
  - Loads configuration from `config.yaml`.
  - Sets up logging.
  - Initializes the market analyzer, risk manager, order executor, and trading strategies.
  - Starts the main trading loop and the FastAPI server.

### 2. Configuration (`config.yaml`)

- **Purpose**: Centralized configuration for the entire system.
- **Sections**:
  - `paper_trade`: Enable/disable paper trading mode.
  - `exchange`: API keys, testnet settings, and other exchange-related parameters.
  - `market_analysis`: Settings for market data fetching and analysis.
  - `execution`: Parameters for order execution, risk management, and position sizing.
  - `strategies`: Configuration for individual trading strategies.
  - `logging`: Logging level and format.

### 3. Strategy Orchestrator (`agents/orchestrator.py`)

- **Purpose**: Manages and activates trading strategies based on the current market regime.
- **Key Class**: `StrategyOrchestrator`
- **Functionality**:
  - Monitors market data from the `MarketAnalysisSystem`.
  - Activates the appropriate strategy (e.g., `BullStrategy`, `BearStrategy`) based on the detected market regime.
  - Generates trade signals and executes them through the `SmartOrderExecutor`.

### 4. API Server (`api/server.py`)

- **Purpose**: Provides a RESTful API and WebSocket interface for interacting with the trading system.
- **Framework**: FastAPI
- **Endpoints**:
  - `/status`: Get the current status of the system.
  - `/performance`: View trading performance metrics.
  - `/risk`: Get information about the current risk exposure.
  - `/ws`: WebSocket endpoint for real-time updates.
  - `/botpress`: Webhook for integration with Botpress.

### 5. Market Analyzer (`core/market_analyzer.py`)

- **Purpose**: Fetches and analyzes market data to identify market regimes and assess risk.
- **Key Classes**: `MarketAnalysisSystem`, `RiskManager`
- **Functionality**:
  - Fetches historical and real-time market data using `ccxt`.
  - Calculates a wide range of technical indicators using `talib`.
  - Detects market regimes (Bullish, Bearish, Ranging, Volatile) using a weighted scoring system.
  - The `RiskManager` assesses the overall market risk and adjusts trading parameters accordingly.

### 6. Exchange Configuration (`core/exchange_config.py`)

- **Purpose**: Manages exchange-specific settings and API credentials.
- **Functionality**:
  - Loads exchange configuration from `config.yaml`.
  - Provides a centralized way to access API keys and other exchange settings.
  - Handles testnet/mainnet configurations.

### 7. Order Executor (`execution/order_executor.py`)

- **Purpose**: Executes trades in a smart and risk-aware manner.
- **Key Class**: `SmartOrderExecutor`
- **Functionality**:
  - Places, cancels, and tracks orders.
  - Supports both live and paper trading modes.
  - Implements risk management rules for position sizing and stop-loss placement.
  - Includes retry logic for handling failed orders.

### 8. Trading Strategies (`strategies/`)

- **Purpose**: Implement the specific logic for generating trade signals in different market conditions.
- **Base Class**: `BaseStrategy` (`strategies/base_strat.py`)
  - Defines the common interface for all strategies.
  - Provides helper methods for data validation and calculating common technical indicators.
- **Concrete Strategies**:
  - `BullStrategy` (`strategies/bull.py`): For bullish market conditions.
  - `BearStrategy` (`strategies/bear.py`): For bearish market conditions.
  - `RangingStrategy` (`strategies/ranging.py`): For ranging markets.
  - `VolatileStrategy` (`strategies/volatile.py`): For volatile markets.
- **Each strategy**:
  - Calculates its own set of technical indicators.
  - Generates entry and exit signals.
  - Defines stop-loss and take-profit levels.
