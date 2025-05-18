"""
Multi-asset ML Trading Bot using LumiBot, FinBERT, and linear regression models.
Features: Sentiment analysis, technical indicators, backtesting, trade logging.
"""
from lumibot.brokers import Alpaca                       # Broker for trade execution and account info
from lumibot.strategies.strategy import Strategy         # Strategy base class for building trading bots
from lumibot.backtesting import YahooDataBacktesting     # Yahoo Finance-based backtesting
from lumibot.traders import Trader                       # Trading event loop
from datetime import datetime, timedelta, timezone       # For time calculations
from dotenv import load_dotenv                           # Loads API keys from .env
import os
import joblib                                            # Model loading
from alpaca.data import NewsClient                       # Alpaca news API
from alpaca.data.requests import NewsRequest             # For building news requests
from finbert_utils import estimate_sentiment             # Sentiment scoring util
from ta.momentum import RSIIndicator                     # RSI calculation
from ta.trend import MACD                                # MACD calculation
from ta.volatility import BollingerBands, AverageTrueRange # Volatility indicators
import pandas as pd

#  Environment Setup 
load_dotenv()
ALPACA_CREDS = {
    "API_KEY": os.getenv('APCA_API_KEY'),
    "API_SECRET": os.getenv('APCA_API_SECRET'),
    "Paper": True
}

# Trade log (will export as .csv at end)
trade_log = []

class MLTrader(Strategy):
    """
    Multi-stock trading strategy using ML regression, FinBERT sentiment, and technical indicators.
    """
    def initialize(self, symbols=None, cash_at_risk: float = 0.5):
        """
        Initializes trading parameters, loads per-stock models, and sets up news client.
        """
        self.symbols = symbols or ["SPY"]
        self.sleeptime = "1D"
        self.cash_at_risk = cash_at_risk
        self.last_trades = {symbol: None for symbol in self.symbols}
        self.news_client = NewsClient(api_key=ALPACA_CREDS["API_KEY"],
                                      secret_key=ALPACA_CREDS["API_SECRET"])
        # Load dictionary of models for all stocks
        self.model_bundle = joblib.load("models/models_per_stock_fd.pkl")     
        self.position_snapshot = {} # Store position at each trade
        # Color mapping for plot markers per symbol
        self.symbol_colors = {
                                "AAPL": "orange",
                                "MSFT": "#DA70D6",
                                "TSLA": "purple",
                                "NVDA": "cyan",
                                "AMZN": "magenta",
                                "GOOGL": "yellow",
                                "META": "#FF7F50",    
                                "SPY": "red"     
                            }


    def log_trade(self, symbol, action, quantity, price, cash_before, cash_after, reason=None, realized_pnl=None):
        """
        Appends trade info to the trade log for export/analysis.
        """
        trade_log.append({
            "datetime": self.get_datetime().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": round(price, 2) if price is not None else None,
            "cash_before": round(cash_before, 2) if cash_before is not None else None,
            "cash_after": round(cash_after, 2) if cash_after is not None else None,
            "realized_pnl": round(realized_pnl, 2) if realized_pnl is not None else None,
            "reason": reason
        })


    def position_sizing(self, symbol):
        """
        Position sizing using ATR-based risk management for each stock.
        Returns: (available_cash, last_price, quantity_to_trade)
        """
        cash=self.get_cash()
        last_price = self.get_last_price(symbol)
        try:
            # Use last 30 days of price data for ATR calculation
            historical_data = self.get_historical_prices(symbol, 30, "day")
            df = historical_data.df
            if not all(col in df.columns for col in ["high", "low", "close"]):
                raise ValueError("Missing expected columns in historical data")
            atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"]).average_true_range().iloc[-1]
            atr = max(atr, last_price * 0.005)
        except Exception as e:
            print(f"ATR Error ({symbol}): {e}")
            atr = last_price * 0.02
        risk_per_share = atr * 2
        max_loss = cash * self.cash_at_risk
        quantity = int(round(max_loss / risk_per_share))
        max_shares = int(cash * 0.3 / last_price)
        quantity = min(max(1, quantity), max_shares)
        return cash, last_price, quantity


    def predict_5d_range(self, symbol):
        """
        Predicts 5-day high/avg/low using per-stock regression models and engineered features.
        Returns: dict with min, avg, max or None if unavailable.
        """
        try:
            model_info = self.model_bundle.get(symbol)
            if not model_info:
                raise ValueError(f"No model found for {symbol}")
            model = model_info["model"]
            predictors = model_info["predictors"]
            df = self.get_historical_prices(symbol, 250, "day").df.reset_index()
            # Feature engineering
            df["Open_Price"] = df["open"]
            df["Close_Ratio_5"] = df["close"] / df["close"].rolling(5).mean()
            df["Trend_5"] = (df["close"].shift(1) < df["close"]).rolling(5).sum()
            df["Close_Ratio_250"] = df["close"] / df["close"].rolling(250).mean()
            df["Trend_250"] = (df["close"].shift(1) < df["close"]).rolling(250).sum()
            df["RSI"] = RSIIndicator(close=df["close"]).rsi()
            df["MACD_diff"] = MACD(close=df["close"]).macd_diff()
            boll = BollingerBands(close=df["close"])
            df["bollinger_pct"] = (df["close"] - boll.bollinger_lband()) / (boll.bollinger_hband() - boll.bollinger_lband())
            df.dropna(inplace=True)
            X_latest = df[predictors].iloc[[-1]]
            pred = model.predict(X_latest)[0]
            return {"min": pred[0], "avg": pred[1], "max": pred[2]}
        except Exception as e:
            print(f"Prediction error ({symbol}): {e}")
            return None


    def get_dates(self):
        """
        Utility for correct date range (live or backtest).
        """
        current_time = self.get_datetime().date() if self.is_backtesting else datetime.now(timezone.utc).date()
        return current_time, current_time - timedelta(days=3)

    def get_sentiment(self, symbol):
        """
        Fetches recent news headlines and estimates sentiment for a stock.
        Returns: (probability, sentiment label)
        """
        try:
            today, three_days_prior = self.get_dates()
            request = NewsRequest(start=three_days_prior, end=today, symbols=symbol, limit=50)
            response = self.news_client.get_news(request)
            articles = response.data["news"]
            headlines = [article.headline for article in articles]
            probability, sentiment = estimate_sentiment(headlines)
            return probability, sentiment
        except Exception as e:
            print(f"News Error ({symbol}): {e}")
            return 0.0, "neutral"

    def on_trading_iteration(self):
        """
        Main loop: executes trading logic for each stock, logs trades, updates plots.
        """
        MAX_TRADES_PER_ITERATION = 3
        trade_count = 0
        available_budget = self.get_cash()
        portfolio_value = self.get_portfolio_value()
        print(f"Portfolio Value: ${portfolio_value:.2f} | Available Cash: ${available_budget:.2f}")
        
        # Step 1: Check for broker-closed positions
        for symbol in self.symbols:
            position = self.get_position(symbol)
            if self.last_trades.get(symbol) == "buy" and not position and symbol in self.position_snapshot:
                snap = self.position_snapshot[symbol]
                self.log_trade(symbol, "auto_close", snap["shares"], snap["avg_price"], snap["cash_before"], self.get_cash())
                print(f"Broker closed long position on {symbol}")
                self.last_trades[symbol] = None
                del self.position_snapshot[symbol]
            elif self.last_trades.get(symbol) == "sell" and not position and symbol in self.position_snapshot:
                snap = self.position_snapshot[symbol]
                self.log_trade(symbol, "auto_close", snap["shares"], snap["avg_price"], snap["cash_before"], self.get_cash())
                print(f"Broker closed short position on {symbol}")
                self.last_trades[symbol] = None
                del self.position_snapshot[symbol]
        self.add_line(name="Portfolio Value", value=portfolio_value, plot_name="Portfolio", color="blue")
        self.add_line(name="Cash", value=available_budget, plot_name="Portfolio", color="green")
        
        # Step 2: Trading logic per symbol
        for symbol in self.symbols:
            if trade_count >= MAX_TRADES_PER_ITERATION:
                print("Max trades for this iteration reached.")
                break

            available_budget, last_price, quantity = self.position_sizing(symbol)
            position = self.get_position(symbol)
            # Plot stock price for all tracked symbols
            self.add_line(
                            name=f"{symbol}_price",
                            value=last_price,
                            plot_name="Stocks",
                            color=self.symbol_colors.get(symbol, "white"),
                        )
            if position:
                print(f"Currently holding {position.quantity} shares of {symbol} at avg. price ${position.avg_fill_price:.2f} → Total: ${position.quantity * position.avg_fill_price:.2f}")

            else:
                print(f"Not holding any position in {symbol}")

            probability, sentiment = self.get_sentiment(symbol)
            range_pred = self.predict_5d_range(symbol)
            if not range_pred:
                self.log_trade(
                    symbol=symbol, action="skipped", quantity=0, price=last_price,
                    cash_before=available_budget, cash_after=available_budget,
                    reason="Model prediction unavailable"
                )
                continue

            print(f"\n{symbol} → Sentiment: {sentiment} ({probability:.2%}), Price: ${last_price:.2f}, Range: {range_pred}")
            if available_budget>last_price:
                # Buy Logic
                if sentiment == "positive" and probability > 0.35 and range_pred["avg"] > last_price:
                    if self.last_trades[symbol] == "sell":
                        cash_before = self.get_cash()
                        realized_pnl = (last_price - position.avg_fill_price) * position.quantity
                        self.sell_all(symbol)
                        self.add_marker(
                                        name=f"{symbol}_close",
                                        value=last_price,
                                        plot_name="Portfolio",
                                        color="white",
                                        symbol="x",
                                        size=8
                                        )
                        self.log_trade(symbol, "close", position.quantity, last_price, cash_before, self.get_cash(), realized_pnl=realized_pnl)
                        print(f"Closed short position on {symbol}")
                        self.last_trades[symbol] = None
                        continue

                    elif not position and self.last_trades[symbol] != "buy":
                        estimated_cost = last_price * quantity
                        print(f"Placing long order: {symbol} | Qty: {quantity} | Cost: ${estimated_cost:.2f}")
                        order = self.create_order(symbol, quantity, "buy", type="bracket",
                                                    take_profit_price=last_price * 1.10,
                                                    stop_loss_price=last_price * 0.97)
                        cash_before = available_budget
                        self.submit_order(order)
                        available_budget -= estimated_cost
                        self.last_trades[symbol] = "buy"
                        trade_count += 1
                        self.log_trade(symbol, "buy", quantity, last_price, cash_before, available_budget)
                        self.position_snapshot[symbol] = {
                                                                "shares": quantity,
                                                                "avg_price": last_price,
                                                                "cash_before": cash_before
                                                        }
                        self.add_marker(
                                        name=f"{symbol}_buy",
                                        value=last_price,
                                        plot_name="Portfolio",
                                        color="green",
                                        symbol="triangle-up",
                                        size=10
                                    )
                    elif not position and range_pred["min"]>last_price* 1.005:
                        estimated_cost = last_price * quantity
                        print(f"Placing long order: {symbol} | Qty: {quantity} | Cost: ${estimated_cost:.2f}")
                        order = self.create_order(symbol, quantity, "buy", type="bracket",
                                                    take_profit_price=last_price * 1.10,
                                                    stop_loss_price=last_price * 0.97)
                        cash_before = available_budget
                        self.submit_order(order)
                        available_budget -= estimated_cost
                        self.last_trades[symbol] = "buy"
                        trade_count += 1
                        self.log_trade(symbol, "buy", quantity, last_price, cash_before, available_budget)
                        self.position_snapshot[symbol] = {
                                                                "shares": quantity,
                                                                "avg_price": last_price,
                                                                "cash_before": cash_before
                                                        }
                        self.add_marker(
                                        name=f"{symbol}_buy",
                                        value=last_price,
                                        plot_name="Portfolio",
                                        color="green",
                                        symbol="triangle-up",
                                        size=10
                                    )

                #Short Logic    
                elif sentiment == "negative" and probability > 0.4 and range_pred["avg"] < last_price:
                    if self.last_trades[symbol] == "buy":
                        cash_before = self.get_cash()
                        realized_pnl = (position.avg_fill_price - last_price) * position.quantity
                        self.sell_all(symbol)
                        self.add_marker(
                                            name=f"{symbol}_close",
                                            value=last_price,
                                            plot_name="Portfolio",
                                            color="white",
                                            symbol="x",
                                            size=8
                                        )
                        self.log_trade(symbol, "close", position.quantity, last_price, cash_before, self.get_cash(), realized_pnl=realized_pnl)
                        print(f"Closed long position on {symbol}")
                        self.last_trades[symbol] = None
                        continue

                    elif not position and probability > 0.95 and self.last_trades[symbol] != "sell":
                        estimated_cost = last_price * (quantity/2)
                        print(f"Placing short order: {symbol} | Qty: {(quantity/2)} | Est. Cost: ${estimated_cost:.2f}")
                        order = self.create_order(
                            symbol, (quantity/2), "sell", type="bracket",
                            take_profit_price=last_price * 0.97,
                            stop_loss_price=last_price * 1.03
                        )
                        cash_before = available_budget
                        self.submit_order(order)
                        available_budget -= estimated_cost
                        self.last_trades[symbol] = "sell"
                        trade_count += 1
                        self.log_trade(symbol, "sell", (quantity/2), last_price, cash_before, available_budget)
                        self.position_snapshot[symbol] = {
                                                            "shares": (quantity/2),
                                                            "avg_price": last_price,
                                                            "cash_before": cash_before
                                                        }
                        self.add_marker(
                                        name=f"{symbol}_sell",
                                        value=last_price,
                                        plot_name="Portfolio",
                                        color="red",
                                        symbol="triangle-down",
                                        size=10
                                    )



# MAIN SCRIPT: Run backtest and save trade log 
if __name__ == "__main__":
    start_date = datetime(2020, 12, 1)
    end_date = datetime(2024, 12, 31)
    symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "SPY"]
    broker = Alpaca(ALPACA_CREDS)
    strategy = MLTrader(name='ml_multi', broker=broker, parameters={"symbols": symbols, "cash_at_risk": 0.5})
    # Execute backtest
    executed_strategy = strategy.backtest(YahooDataBacktesting, start_date, end_date, parameters={"symbols": symbols, "cash_at_risk": 0.5})

    # Export trade log to CSV
    try:
        pd.DataFrame(trade_log).to_csv("trade_ledger.csv", index=False)
        print("✅ Trade ledger saved to trade_ledger.csv")
    except AttributeError as e:
        print(f"⚠️ Trade log not found: {e}")
