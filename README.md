# TradingBot Assignment

## Overview
This project implements a multi-stock trading bot using LumiBot and FinBERT for sentiment analysis. The bot also uses a pre-trained linear regression model to predict 5-day price ranges.

---

## Directory Structure
├── trading_bot_multistock.py # Main trading script
├── finbert_utils.py # Sentiment analysis functions
├── models/ # Trained models
├── data_example/ # Example SP500 data
├── research/ # Exploratory scripts & notebooks
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

---

## Roadmap & Future Work

- Add REST API (FastAPI) for live prediction/trading.
- Integrate additional ML models.
- Improve risk management and reporting.
- Live trading with brokerage APIs.

---

## Author

Alan Varghese Vinu  
(Your email here)


