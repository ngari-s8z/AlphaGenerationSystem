import time
import numpy as np
import pandas as pd
from src.alpha_pipeline import AlphaPipeline
from broker_api import execute_trade


# Initialize pipeline with live data
pipeline = AlphaPipeline(
    transformer_model_path="models/optimized_alpha_transformer.pth",
    rl_model_path="models/ppo_alpha_optimizer.zip",
    market_data_path="data/WorldQuant_Financial_Datasets.csv",
    alpha_data_path="data/alpha50.csv",
    use_live_data=True
)

class LiveAlphaExecutor:
    """Automated live trading system for Alpha execution."""

    def __init__(self, alpha_expr, trade_interval=60):
        self.alpha_expr = alpha_expr
        self.trade_interval = trade_interval  # Frequency of trade execution (seconds)
        self.last_positions = {}

    def execute_live_trades(self):
        """Fetches real-time data and executes trades dynamically."""
        while True:
            live_results = pipeline.backtest_alpha(self.alpha_expr)

            if live_results:
                df_alpha = pipeline.backtest_alpha(self.alpha_expr)
                df_alpha["position"] = np.where(df_alpha["alpha_rank"] > 0.5, "BUY", "SELL")

                for asset, signal in df_alpha["position"].items():
                    if asset in self.last_positions and self.last_positions[asset] == signal:
                        continue  # Avoid redundant trades

                    execute_trade(asset, signal)  # Execute trade via broker API
                    self.last_positions[asset] = signal

            time.sleep(self.trade_interval)

# Example usage
if __name__ == "__main__":
    alpha_expr = "rank(close) - ts_mean(volume, 10)"
    live_executor = LiveAlphaExecutor(alpha_expr)
    live_executor.execute_live_trades()
