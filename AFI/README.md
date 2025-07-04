# AFI Trading System

## Overview

The AFI trading system is a sophisticated, automated cryptocurrency trading bot designed to analyze market conditions, execute trades, and manage risk with a high degree of autonomy. It features a modular architecture that separates concerns, allowing for easy extension and customization of its components.

At its core, the system identifies the prevailing market regime (Bullish, Bearish, Ranging, or Volatile) and dynamically activates the most suitable trading strategy. This allows it to adapt to changing market conditions and optimize its trading performance.

## Features

- **Dynamic Strategy Orchestration**: Automatically selects and activates the best trading strategy based on real-time market analysis.
- **Multi-Regime Analysis**: Identifies Bullish, Bearish, Ranging, and Volatile market conditions to inform trading decisions.
- **Advanced Risk Management**: Integrated risk management system to control position sizing, set stop-loss orders, and manage overall portfolio risk.
- **Paper Trading Mode**: A simulation mode to test strategies and system performance without risking real capital.
- **REST API and WebSockets**: A comprehensive API for interacting with the system, along with a WebSocket interface for real-time updates.
- **Extensible Architecture**: Easily add new trading strategies, indicators, or other components.

## System Architecture

For a detailed look at the project's structure and components, please see the following documents:

- **[Directory Structure (`docs/structure.md`)](docs/structure.md)**
- **[Component Documentation (`docs/documentation.md`)](docs/documentation.md)**

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/AFI.git
    cd AFI
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the System**:
    -   Create a `config.yaml` file based on the structure outlined in `docs/documentation.md` or by copying `config.example.yaml` if it exists.
    -   Add your exchange API keys (for both testnet and mainnet), Botpress secret token, and other required settings.

## Usage

🛠️ Binance API Setup

To use the AFI Trading System with Binance, you must generate API keys. You can run the system in two modes:

    ✅ Paper Trading (Testnet): for simulation only, no real money involved.

    ⚠️ Live Trading (Mainnet): executes real trades with real funds.

✅ Testnet Setup (Safe Paper Trading)

Use this mode to test everything without risking real money.
Step-by-Step: Create Testnet API Keys

    Go to the Binance Testnet
    👉 https://testnet.binancefuture.com/

    Log in with GitHub (or create an account)

    Generate API Keys

        Go to "API Key" section.

        Click "Create" and give your key a label (e.g., afi-testnet).

        Copy the API Key and Secret Key.

    Fund Your Testnet Wallet

        Visit the Futures Testnet Wallet

        Click "Transfer" → Add USDT to your test account.

    Update config.yaml:

  exchange:
    name: 'binance'
    testnet: true               # ✅ Must be true
    mainnet_fallback: false     # ✅ Must be false
    testnet_api:
      key: 'your_testnet_api_key_here'
      secret: 'your_testnet_secret_here'

  trading:
    paper_trading: true         # ✅ Enables paper trading

    

⚠️ Live Trading with Real Funds (Mainnet)

    WARNING: Live trading uses real money. Proceed only if you understand the risks. Make sure paper trading works flawlessly before enabling mainnet trading.

Step-by-Step Guide: Creating Binance Mainnet API Keys

    Create or Log in to Binance
    👉 https://www.binance.com/

    Enable Two-Factor Authentication (2FA)

        Go to Security Settings

        Enable Google Authenticator or SMS verification

    Create API Keys

        Go to API Management

        Create a new key (e.g., afi-live)

        Complete 2FA verification and email confirmation

        Save the API Key and Secret Key (you won't see the secret again)

    Configure API Permissions

✅ Enable:

    ✓ Enable Spot & Margin Trading

    ✓ Enable Reading

🚫 DO NOT ENABLE:

    ✗ Enable Withdrawals

🔐 (Recommended): Restrict IPs to your server/machine for added security.

Update config.yaml:

  exchange:
    name: 'binance'
    testnet: false              # ⚠️ Must be false
    mainnet_fallback: true      # ⚠️ Must be true
    mainnet_api:
      key: 'your_mainnet_api_key_here'
      secret: 'your_mainnet_secret_here'

  trading:
    paper_trading: false        # ⚠️ Must be false

### Running the System

-   **Live Trading**:
    ```bash
    python main.py
    ```

-   **Paper Trading**:
    ```bash
    python run_paper_trade.py
    ```

### API Endpoints

The API server provides several endpoints to monitor and interact with the system:

-   `GET /status`: Returns the current status of the trading system, including the active market regime and strategy.
-   `GET /performance`: Provides a summary of trading performance metrics.
-   `GET /risk`: Shows the current risk exposure and other risk-related information.
-   `POST /botpress`: Webhook for receiving commands from a Botpress chatbot.
-   `WS /ws`: WebSocket endpoint for receiving real-time updates.

### Botpress Commands

If you have integrated the system with Botpress, you can use the following commands:

-   `/status`: Get a summary of the system's current status.
-   `/performance`: Retrieve performance metrics.
-   `/risk`: Get a risk report.
-   `/help`: See a list of available commands.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to the repository.
