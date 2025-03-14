import os
import pandas as pd
import requests
import time
import io  # ✅ Fix for AttributeError
from datetime import datetime, timedelta

# Alpha Vantage API Key (replace with your actual key)
ALPHA_VANTAGE_API_KEY = "7Z2ALELTK1NJE6XY"

def get_ticker_list():
    """Returns a comprehensive list of tickers covering S&P 500, NASDAQ 100, and global markets."""
    return [
        # S&P 500 Top Companies
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "PYPL", "ADBE",
        # "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "MA", "HD", "DIS", "KO",
        # "PEP", "XOM", "CSCO", "PFE", "T", "MRK", "ABT", "INTC", "VZ", "NKE",
        # "LLY", "MCD", "MDT", "ABBV", "DHR", "WMT", "CRM", "TXN", "COST", "TMO",
        # "AVGO", "ACN", "LIN", "BMY", "CVX", "QCOM", "AMD", "NOW", "HON",

        # # NASDAQ 100 Top Companies
        # "GOOG", "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "NFLX", "META", "ADBE", "PYPL",
        # "CSX", "ILMN", "INTU", "MDLZ", "BKNG", "MELI", "JD", "ATVI", "LRCX", "KDP",
        # "ADP", "CME", "CDNS", "PDD", "XEL", "IDXX", "KLAC", "VRSN", "WBA", "CTAS",

        # # Dow Jones 30 Companies
        # "MMM", "AXP", "BA", "CAT", "CVX", "CSCO", "KO", "XOM", "GS", "HD",
        # "IBM", "INTC", "JNJ", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV",
        # "UNH", "VZ", "V", "WMT", "DIS", "AAPL", "CRM", "DOW", "AMGN", "HON"
    ]

def fetch_alpha_vantage_data(ticker):
    """Fetches historical stock data for a given ticker from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "full",  # Get full historical data
        "datatype": "csv"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))  # ✅ Fixed AttributeError
    else:
        print(f"⚠️ Failed to fetch data for {ticker}")
        return None

def process_and_save_data():
    """Fetches historical market data for all tickers and saves it to CSV."""
    tickers = get_ticker_list()
    all_data = []

    for i, ticker in enumerate(tickers):
        print(f"📡 Fetching data for {ticker} ({i+1}/{len(tickers)})...")
        df = fetch_alpha_vantage_data(ticker)

        if df is not None:
            df["Ticker"] = ticker
            all_data.append(df)

        time.sleep(12)  # Alpha Vantage free-tier allows 5 requests per minute

    if all_data:
        final_df = pd.concat(all_data)
        final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
        yesterday = datetime.now() - timedelta(days=1)
        final_df = final_df[final_df["timestamp"] <= yesterday]

        os.makedirs("data", exist_ok=True)
        final_df.to_csv("data/historical_market_data.csv", index=False)
        print("✅ Historical market data saved successfully!")

# Run script
if __name__ == "__main__":
    process_and_save_data()
