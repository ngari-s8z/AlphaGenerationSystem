import yfinance as yf
import pandas as pd
import time

class LiveMarketData:
    """Fetches live market data for real-time Alpha execution."""
    
    def __init__(self, tickers=["AAPL", "GOOGL", "MSFT"], interval="1m"):
        self.tickers = tickers
        self.interval = interval

    def fetch_live_data(self):
        """Fetches latest market data for defined tickers."""
        data = {}
        for ticker in self.tickers:
            df = yf.download(ticker, period="1d", interval=self.interval)
            data[ticker] = df.iloc[-1]  # Get the most recent data point
        return pd.DataFrame(data)

    def stream_data(self, update_interval=60):
        """Continuously fetches live market data at specified intervals."""
        while True:
            live_data = self.fetch_live_data()
            print("Live Market Data:\n", live_data)
            time.sleep(update_interval)

# Example usage
if __name__ == "__main__":
    live_data_streamer = LiveMarketData()
    live_data_streamer.stream_data()
