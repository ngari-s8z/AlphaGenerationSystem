import pandas as pd
import numpy as np
from live_market_data import LiveMarketData

class AdvancedAlphaBacktester:
    """Multi-asset backtesting engine with real-time execution & risk management."""

    def __init__(self, df_market_data, df_alphas, live_data_source=None, initial_capital=1000000):
        self.df_market_data = df_market_data
        self.df_alphas = df_alphas
        self.live_data_source = live_data_source or LiveMarketData()
        self.initial_capital = initial_capital
        self.results = None

    def apply_alpha(self, alpha_expr, df=None):
        """Computes Alpha signals dynamically over market data."""
        df = df if df is not None else self.df_market_data.copy()
        try:
            df["alpha_signal"] = eval(alpha_expr, {}, df.to_dict("series"))
            df["alpha_rank"] = df.groupby("date")["alpha_signal"].rank(pct=True)
            return df
        except Exception as e:
            print(f"Error evaluating Alpha expression: {e}")
            return None

    def compute_metrics(self, df, max_leverage=2, stop_loss=0.05):
        """Calculates performance metrics with risk control."""
        df["returns"] = df.groupby("asset")["close"].pct_change()
        df["position"] = np.where(df["alpha_rank"] > 0.5, 1, -1)  # Long top 50%, short bottom 50%

        # Adjust position for risk control
        df["position"] = np.clip(df["position"], -max_leverage, max_leverage)

        df["strategy_returns"] = df["position"].shift(1) * df["returns"]

        # Apply stop-loss
        df.loc[df["strategy_returns"] < -stop_loss, "strategy_returns"] = -stop_loss

        sharpe_ratio = df["strategy_returns"].mean() / df["strategy_returns"].std()
        max_drawdown = (df["strategy_returns"].cummax() - df["strategy_returns"]).max()
        turnover = df["position"].diff().abs().mean()

        return {
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Turnover": turnover,
            "Cumulative Returns": df["strategy_returns"].cumsum().iloc[-1]
        }

    def backtest_alpha(self, alpha_expr):
        """Runs backtest for a given Alpha expression."""
        df_alpha = self.apply_alpha(alpha_expr)
        if df_alpha is None:
            return None
        return self.compute_metrics(df_alpha)

    def execute_live_alpha(self, alpha_expr):
        """Runs real-time execution of an Alpha using live market data."""
        live_market_df = self.live_data_source.fetch_live_data()
        df_alpha = self.apply_alpha(alpha_expr, df=live_market_df)
        return self.compute_metrics(df_alpha) if df_alpha is not None else None

    def run_backtest_suite(self):
        """Backtests multiple Alpha expressions and saves results."""
        results = []
        for _, row in self.df_alphas.iterrows():
            alpha_expr = row["formula"]
            metrics = self.backtest_alpha(alpha_expr)
            if metrics:
                results.append({"Alpha": alpha_expr, **metrics})
        self.results = pd.DataFrame(results)
        return self.results

# Load market & Alpha datasets
df_market_data = pd.read_csv("data/WorldQuant_Financial_Datasets.csv")
df_alphas = pd.read_csv("data/alpha50.csv")

# Run backtesting engine
backtester = AdvancedAlphaBacktester(df_market_data, df_alphas)
backtest_results = backtester.run_backtest_suite()
print(backtest_results)

# Save results
backtest_results.to_csv("results/Backtesting_Results.csv", index=False)
