from fastapi import FastAPI, HTTPException      #pip install fastapi
from pydantic import BaseModel
import joblib
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import sys, os
sys.path.append(os.path.abspath(".."))
from finbert_utils import estimate_sentiment
import pandas as pd

# FastAPI Setup
app = FastAPI(title="TradingBot Prediction API")

# Model and Feature Configuration
MODEL_PATH = "../models/linear_model_5d_price.pkl"
# Load model and predictors once on startup
try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    predictors = bundle["predictors"]
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# Request Schema

class PredictRequest(BaseModel):
    symbol: str
    
class SentimentRequest(BaseModel):
    headlines: list[str]

# Routes

@app.post("/predict")
def predict_5d_range(req: PredictRequest):
    """
    Predicts the 5-day price range (min, avg, max) for a given stock symbol.
    """
    symbol = req.symbol.upper()
    try:
        # Download historical daily data for the last 400 days
        df = yf.Ticker(symbol).history(period="400d")
        df.index = pd.to_datetime(df.index, utc=True)
        df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors='ignore')
        if len(df) < 250:
            raise ValueError("Not enough historical data for this symbol.")
        # Feature Engineering
        df = df.reset_index()
        df["Open_Price"] = df["Open"]
        print(df)
        df["Close_Ratio_5"] = df["Close"] / df["Close"].rolling(5).mean()
        df["Trend_5"] = (df["Close"].shift(1) < df["Close"]).rolling(5).sum()
        df["Close_Ratio_250"] = df["Close"] / df["Close"].rolling(250).mean()
        df["Trend_250"] = (df["Close"].shift(1) < df["Close"]).rolling(250).sum()
        df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
        macd = MACD(close=df["Close"])
        df["MACD_diff"] = macd.macd_diff()
        boll = BollingerBands(close=df["Close"])
        df["bollinger_pct"] = (
            (df["Close"] - boll.bollinger_lband()) /
            (boll.bollinger_hband() - boll.bollinger_lband())
        )
        df = df.dropna()
        # Prepare the latest data point for prediction
        print(df)
        X_latest = df[predictors].iloc[[-1]]
        pred = model.predict(X_latest)[0]
        return {"symbol": symbol, "min": float(pred[0]), "avg": float(pred[1]), "max": float(pred[2])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@app.post("/sentiment")
def predict_sentiment(req: SentimentRequest):
    """
    Estimates sentiment (positive/negative/neutral) and probability for a list of news headlines.
    """
    try:
        probability, sentiment = estimate_sentiment(req.headlines)
        return {"probability": probability, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sentiment error: {e}")
#pip install uvicorn
#uvicorn app_demo:app --reload
# Access at:
# http://127.0.0.1:8000/docs (Swagger UI)
