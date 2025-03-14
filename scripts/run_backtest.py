import pandas as pd
from src.backtesting_engine import AdvancedAlphaBacktester

# Load datasets
df_market_data = pd.read_csv("data/WorldQuant_Financial_Datasets.csv")
df_alphas = pd.read_csv("data/alpha50.csv")

# Initialize backtesting engine
backtester = AdvancedAlphaBacktester(df_market_data, df_alphas)

# Run backtesting
backtest_results = backtester.run_backtest_suite()

# Save results
backtest_results.to_csv("results/Backtesting_Results.csv", index=False)
print("Backtesting complete. Results saved.")
