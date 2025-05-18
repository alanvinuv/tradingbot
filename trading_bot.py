# Importing Required Libraries
from lumibot.brokers import Alpaca                          # Broker interface for executing trades and retrieving account info
from lumibot.strategies.strategy import Strategy            # Base class for creating trading strategies in LumiBot
from lumibot.backtesting import YahooDataBacktesting        # Backtesting module to simulate trading using historical Yahoo Finance data
from lumibot.traders import Trader                          # Trader class to run strategies, manage event loop and scheduling [Paper/Live Trading]
from datetime import datetime, timedelta, timezone          # Standard Python module for working with dates and time zones
from dotenv import load_dotenv                              # Loads environment variables from a .env file for API keys and secrets
import os                                                   # Standard module for OS operations, e.g., reading environment variables

from alpaca.data import NewsClient
from alpaca.data.requests import NewsRequest                # Alpaca API modules for fetching financial news, market data, and sending requests

from finbert_utils import estimate_sentiment                # Custom module: Accessing FinBERT sentiment analysis utilities

import joblib                                               # Utility for saving/loading pre-trained machine learning models

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange  # TA-Lib: Technical analysis indicators for financial data
# RSI (Relative Strength Index): RSI is a momentum oscillator that measures the speed and change of recent price movements.
# MACD (Moving Average Convergence Divergence): MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
# ATR (Average True Range): ATR measures market volatility by decomposing the entire range of an asset price for that period.
# Bollinger Bands: Bollinger Bands consist of a moving average (middle band) and two standard deviation lines (upper and lower bands).

load_dotenv()       # Load environment variables from .env (API keys, etc.)

# Prepare Alpaca credentials from environment
ALPACA_CREDS={
    "API_KEY" : os.getenv('APCA_API_KEY'),
    "API_SECRET" : os.getenv('APCA_API_SECRET'),
    "Paper" : True
}

class MLTrader(Strategy):       
    """A strategy class for ML-driven trading with sentiment and regression predictions."""
    
    def initialize(self, symbol:str="SPY", cash_at_risk:float=0.5):  
        """ Initializing Parameters and trading on the SPY index S&P 500 ETF Trust """
        
        self.symbol =symbol
        self.sleeptime = "1D"       # Run once daily
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        # Initialize Alpaca NewsClient for news sentiment
        self.news_client = NewsClient(api_key=ALPACA_CREDS["API_KEY"],
                                      secret_key=ALPACA_CREDS["API_SECRET"])
        # Load pre-trained regression model and its predictors
        bundle = joblib.load("models/linear_model_5d_price.pkl")
        self.model = bundle["model"]
        self.predictors = bundle["predictors"]


    def predict_5d_range(self):
        """
        Predicts the 5-day price range using the regression model and engineered features.
        Returns dict with min/avg/max predicted prices or None on failure.
        """
        try:
            df = self.get_historical_prices(self.symbol, 250, "day").df.reset_index()

            # Feature engineering (For the model predictors)
            df["Open_Price"] = df["open"]
            df["Close_Ratio_5"] = df["close"] / df["close"].rolling(5).mean()
            df["Trend_5"] = (df["close"].shift(1) < df["close"]).rolling(5).sum()
            df["Close_Ratio_250"] = df["close"] / df["close"].rolling(250).mean()
            df["Trend_250"] = (df["close"].shift(1) < df["close"]).rolling(250).sum()

            df["RSI"] = RSIIndicator(close=df["close"]).rsi()
            macd = MACD(close=df["close"])
            df["MACD_diff"] = macd.macd_diff()
            boll = BollingerBands(close=df["close"])
            df["bollinger_pct"] = (df["close"] - boll.bollinger_lband()) / (boll.bollinger_hband() - boll.bollinger_lband())

            df.dropna(inplace=True)
            # Predict based on latest data
            X_latest = df[self.predictors].iloc[[-1]]  
            pred = self.model.predict(X_latest)[0]

            return {"min": pred[0], "avg": pred[1], "max": pred[2]}

        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    
    def position_sizing(self):
        """
        Determines position size using ATR-based risk management.
        Returns: (cash, last_price, quantity)
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)

        try:
            # Fetch recent price data for ATR calculation
            # Get historical prices using Lumibot's internal method
            historical_data = self.get_historical_prices(self.symbol, 30, "day")
            df = historical_data.df  # Extract DataFrame
            
            # Ensure necessary columns are present
            if not all(col in df.columns for col in ["high", "low", "close"]):
                raise ValueError("Missing expected columns in historical data")

            atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range().iloc[-1]
            atr = max(atr, last_price * 0.02)

        except Exception as e:
            print(f"ATR Error (fallback to static %): {e}")
            atr = last_price * 0.02

        # Calculate position size
        # Risk per share and max loss per trade
        risk_per_share = atr * 1.5
        max_loss = cash * self.cash_at_risk
        quantity = int(round(max_loss / risk_per_share))
        # Position limits
        max_shares = int(cash * 0.3 / last_price)
        quantity = min(max(1, quantity), max_shares)
        print(f"\nATR Sizing: Price=${last_price:.2f}, ATR=${atr:.2f}, Quantity={quantity}")
        return cash, last_price, quantity


    def get_dates(self):
        """
        Returns today's date and the date 3 days prior, for both backtesting and live.
        """
        if self.is_backtesting:
            current_time = self.get_datetime().date() 
        else:
            current_time = datetime.now(timezone.utc).date()

        three_days_prior = current_time - timedelta(days=3)
        return current_time, three_days_prior

    
    def get_sentiment(self):
        """
        Fetches recent news and computes sentiment using FinBERT.
        Returns: (probability, sentiment_label) or empty tuple on error.
        """
        try:
            today, three_days_prior = self.get_dates()
            request = NewsRequest(
                start=three_days_prior,
                end=today,
                symbols=self.symbol,
                limit=50
            )
            response = self.news_client.get_news(request)
            articles = response.data["news"]
            headlines = [article.headline for article in articles]
            probability, sentiment = estimate_sentiment(headlines)
            return probability, sentiment

        except Exception as e:
            print(f"\nNews Error: {e}")
            return []
   

    def on_trading_iteration(self):
        """
        Called on each trading loop iteration: determines position, checks signals, and submits orders.
        """
        cash, last_price, quantity = self.position_sizing()
        position = self.get_position(self.symbol)
        probability, sentiment = self.get_sentiment()
        range_pred = self.predict_5d_range()
        if range_pred:
            print(f"Predicted 5D Range: Min={range_pred['min']:.2f}, Avg={range_pred['avg']:.2f}, Max={range_pred['max']:.2f}")
        print(f"\nSentiment: {sentiment} ({probability:.2%}) | Price: ${last_price:.2f} | Cash: ${cash:.2f}")
        print(f"\nPosition:{position}")
        
        #  Trade logic: Buy/Sell based on combined sentiment and price prediction 
        if cash > last_price:
            # Buy logic
            if sentiment == "positive" and probability > 0.35 and range_pred["avg"] > last_price:   # If sentiment is positive, with decent probability, and model predicts incline
                if self.last_trade == "sell":   # If short, close it
                    self.sell_all()
                    self.last_trade = None
                elif not position and self.last_trade != "buy":     # If no position, and we didn’t just try to buy, open long
                    print("Strong buy signal triggered.")
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.10,
                        stop_loss_price=last_price * 0.97
                    )
                    self.submit_order(order)
                    self.last_trade = "buy"  
                elif not position and range_pred["min"]>last_price:     # If still no position (e.g., buy didn’t go through), but model is strongly likely to go up, invest again
                    print("Weak buy signal based on predicted min > price.")
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "buy",
                        type="bracket",
                        take_profit_price=last_price * 1.10,
                        stop_loss_price=last_price * 0.97
                    )
                    self.submit_order(order)
                    self.last_trade = "buy"
                    
            # Sell logic
            elif sentiment == "negative" and probability > 0.4 and range_pred["avg"] < last_price:      # If sentiment is negative, probability is strong, and model predicts decline
                if self.last_trade == "buy":                 # If we were previously long, exit the position
                    self.sell_all()
                    self.last_trade = None
                elif not position and probability > 0.9 and self.last_trade != "sell":          # If no current position, sentiment is very strong, and last trade wasn't a short — open a short position
                    print("Strong short signal triggered.")
                    order = self.create_order(
                        self.symbol,
                        quantity,
                        "sell",
                        type="bracket",
                        take_profit_price=last_price * 0.95,        # 10% profit target
                        stop_loss_price=last_price * 1.03           # 3% stop loss
                    )
                    self.submit_order(order)
                    self.last_trade = "sell"
            # Neutral Sentiment or no supporting Prediction        
            else:
                print("No actionable sentiment/prediction — skipping.")



# MAIN: Backtest over defined period 
if __name__ == "__main__":
    # Set backtest period
    start_date = datetime(2020, 12, 1)
    end_date = datetime(2024, 12, 31)

    # Initialize broker (paper trading) and strategy
    broker = Alpaca(ALPACA_CREDS)
    strategy = MLTrader(
        name='mlstrat', broker=broker,
        parameters={"symbol": "SPY", "cash_at_risk": 0.5}
    )

    # Run backtest with Yahoo Finance data
    strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={"symbol": "SPY", "cash_at_risk": 0.5}
    )



