import pandas as pd
import time

class TradeLogger:
    """Monitors and logs executed trades in real-time."""
    
    def __init__(self, log_file="logs/trade_log.csv"):
        self.log_file = log_file

    def log_trade(self, asset, signal, price, timestamp):
        """Logs executed trades to a CSV file."""
        df = pd.DataFrame([[asset, signal, price, timestamp]], columns=["Asset", "Signal", "Price", "Timestamp"])
        df.to_csv(self.log_file, mode='a', header=not pd.io.common.file_exists(self.log_file), index=False)
        print(f"Logged Trade: {asset} | {signal} | {price} | {timestamp}")

# Example usage
if __name__ == "__main__":
    logger = TradeLogger()
    logger.log_trade("AAPL", "BUY", 150.25, time.strftime("%Y-%m-%d %H:%M:%S"))
