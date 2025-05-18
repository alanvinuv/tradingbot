# TradingBot Assignment

## Overview
This project implements a multi-stock trading bot using LumiBot and FinBERT for sentiment analysis. The bot also uses a pre-trained linear regression model to predict 5-day price ranges.

---

## Directory Structure
├── trading_bot_multistock.py # Main trading script
├── finbert_utils.py # Sentiment analysis functions
├── models/ # Trained models
├── data_example/ # Example SP500 data
├── research/ # Exploratory scripts & notebooks and prototypes
├── trade_ledger.csv # Sample trade log
├── requirements.txt # Python dependencies
├── .env.example # Example environment variables
├── tests/ # Simple tests
└── README.md

---

## Setup Instructions

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

---

## Model Files

- The `/models` folder contains pre-trained models used by the bot.
- If retraining is needed, see `research/Prediction_model.ipynb`.

---


---

## Notes

- **No live trading is enabled by default**. This is a backtesting framework.
- See `trade_ledger.csv` for a sample trade log output.

endpoint_demo.py demonstrates how the model and sentiment logic can be exposed as an API endpoint using FastAPI for integration and extensibility.
In your research notebook or a section of the README, show how to start the server and make a sample request using curl or Swagger UI.
---


---



# Multi-Asset ML Trading Bot

## Overview

**Multi-Asset ML Trading Bot** is a modular Python framework for automated backtesting and multi-stock trading.  
It uses LumiBot for trade simulation and execution, machine learning models for price forecasting, and FinBERT for news sentiment analysis, enhanced with technical indicators for robust decision-making.

All trades are logged and saved as CSV for further evaluation and analysis.

---

## Project Structure

├── trading_bot_multistock.py # Main trading script
├── finbert_utils.py # Sentiment analysis functions
├── models/ # Trained ML models
│ └── models_per_stock_fd.pkl
├── data_example/ # Example SP500 data
├── research/ # Notebooks & prototypes
│ └── Prediction_model.ipynb
├── trade_ledger.csv # Trade log output (generated)
├── requirements.txt # Python dependencies
├── .env.example # Example for environment variables
├── tests/ # Simple tests
└── README.md # This file

---

## Libraries

lumibot
ta
joblib
python-dotenv
pandas
finbert-embedding # or your finbert_utils.py requirements
alpaca-py
yfinance
scikit-learn
matplotlib


#### Install with:

'''bash
pip install -r requirements.txt


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

