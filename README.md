# Multi-Asset ML Trading Bot


This algorithmic trading framework combines machine learning and sentiment analysis for multi-asset trading strategies.

---

## Overview

The Trading Bot uses LumiBot for trade simulation and execution, machine learning models for price forecasting, machine learning models multioutput regressor(Linear regression) to predict 5-day price ranges. , and FinBERT for news sentiment analysis. All trades are logged and saved as CSV for further evaluation and analysis.

---

## Project Structure

```markdown 
tradingbot/
├── data_example/               # Example SP500 data
│   └── sp500.csv
│
├── models/ # Trained ML models
│   ├── linear_model_5d_price.pkl
│   └── models_per_stock_fd.pkl
│
├── outputs/                    # Outputs of Single and Multistock bot
│   ├── multi_stock      
│   └── single_stock
│ 
├── research/                   # Notebooks & prototypes
│ ├── Prediction_model.ipynb
│ └── app_demo.py
│
├── .env.example                # Example for environment variables
├── requirements.txt            # Python dependencies
├── finbert_utils.py            # Sentiment analysis functions
├── trading_bot_multistock.py   # Main multi-stock trading script
├── trade_ledger.csv            # Trade log output 
├── trading_bot.py              # single-stock trading script
└── README.md # This file
```
**Model Files Explained**

- The `/models` folder contains pre-trained models used by the bot.
- If retraining is needed, see `research/Prediction_model.ipynb`.
---

## Libraries
- lumibot
- ta
- joblib
- python-dotenv
- pandas
- finbert-embedding
- alpaca-py
- yfinance
- scikit-learn
- matplotlib
- transformers
- torch
- numpy
---

### Setup Instructions

1. **Clone the repo**
git clone <your-repo-link>
cd TradingBot-Assignment


2. **Set up the environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. **Set up your `.env` file**
- Copy `.env.example` to `.env` and fill in your API keys, if needed.

4. **Run the trading bot**
    python trading_bot_multistock.py

- See inline comments for configuration options.


#### Install with:

'''bash
pip install -r requirements.txt

##


## Setup Instructions
#### 1. Clone the repository:

```bash
git clone <your-repo-link>
cd TradingBot-Assignment
```

#### 2. Set up the Python environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Configure your .env file:

Copy .env.example to .env

Fill in your Alpaca API credentials:

```bash
APCA_API_KEY=your_key_here
APCA_API_SECRET=your_secret_here
```

#### 4. Run the trading bot:

```bash
python trading_bot_multistock.py
```

Backtesting results and trade logs are saved as trade_ledger.csv.

## Architecture

### Core Trading Engine
- **LumiBot Integration** for both backtesting and live trade execution
- Multi-asset portfolio support (equities, ETFs)
- Event-driven architecture with real-time capability

### Machine Learning Pipeline
- **Price Forecasting Model**:
  - Multi-output linear regression model
  - Predicts 5-day high/low price ranges
  - Trained on OHLCV + technical indicators
  - Model persistence: `models/models_per_stock_fd.pkl`

### Sentiment Analysis Module
- **FinBERT Implementation**:
  - Financial-domain specific NLP transformer
  - Processes news headlines and earnings reports
  - Generates sentiment scores (negative/neutral/positive)
  - Integrated via `finbert_utils.py`

### Technical Framework
- Feature Engineering:
  - 20+ technical indicators (RSI, MACD, Bollinger Bands)
  - Rolling window feature generation
- Risk Management:
  - Dynamic position sizing
  - Stop-loss/take-profit triggers


## Backtesting

## Notes

- **No live trading is enabled by default**. This is a backtesting framework.
- See `trade_ledger.csv` for a sample trade log output.

app_demo.py demonstrates how the model and sentiment logic can be exposed as an API endpoint using FastAPI for integration and extensibility.
In your research notebook or a section of the README, show how to start the server and make a sample request using curl or Swagger UI.

## Future Works

## Credits