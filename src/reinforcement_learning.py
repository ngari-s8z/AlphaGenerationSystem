import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from gym import spaces

class AlphaTradingEnv(gym.Env):
    """
    Custom environment for Reinforcement Learning-based Alpha Optimization.
    The agent selects Alpha strategies to maximize Sharpe ratio while considering risks.
    """

    def __init__(self, df_market_data, df_alphas, max_steps=1000, initial_capital=1e6):
        super(AlphaTradingEnv, self).__init__()

        # Load market and Alpha data
        self.df_market_data = df_market_data
        self.df_alphas = df_alphas
        self.initial_capital = initial_capital
        self.max_steps = max_steps

        # Define state and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df_market_data.shape[1],), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(df_alphas))

        # Tracking variables
        self.current_step = 0
        self.alpha_history = []
        self.portfolio_value = initial_capital

    def step(self, action):
        """
        Executes the chosen Alpha strategy and calculates reward based on performance.
        """
        alpha_expr = self.df_alphas.iloc[action]["formula"]
        df_alpha = self.apply_alpha(alpha_expr)

        if df_alpha is None:
            return self._get_observation(), -1, True, {}

        # Compute reward using risk-adjusted Sharpe Ratio
        sharpe_ratio, drawdown, turnover = self.compute_metrics(df_alpha)
        reward = self.calculate_reward(sharpe_ratio, drawdown, turnover)

        # Update state
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}

    def reset(self):
        """
        Resets environment state for a new training episode.
        """
        self.current_step = 0
        self.alpha_history = []
        self.portfolio_value = self.initial_capital
        return self._get_observation()

    def _get_observation(self):
        """
        Retrieves the current market state as the RL model's observation.
        """
        return self.df_market_data.iloc[self.current_step].values

    def apply_alpha(self, alpha_expr):
        """
        Computes Alpha signals dynamically over market data.
        """
        try:
            df = self.df_market_data.copy()
            df["alpha_signal"] = eval(alpha_expr, {}, df.to_dict("series"))
            df["alpha_rank"] = df.groupby("date")["alpha_signal"].rank(pct=True)
            return df
        except Exception as e:
            print(f"Error evaluating Alpha expression: {e}")
            return None

    def compute_metrics(self, df):
        """
        Calculates Sharpe ratio, drawdown, and turnover.
        """
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["alpha_rank"].shift(1) * df["returns"]

        sharpe_ratio = df["strategy_returns"].mean() / df["strategy_returns"].std()
        drawdown = (df["strategy_returns"].cummax() - df["strategy_returns"]).max()
        turnover = df["alpha_rank"].diff().abs().mean()

        return sharpe_ratio, drawdown, turnover

    def calculate_reward(self, sharpe, drawdown, turnover):
        """
        Computes a risk-adjusted reward function.
        """
        risk_penalty = drawdown * 0.5 + turnover * 0.3
        return sharpe - risk_penalty

# Load market and Alpha datasets
df_market_data = pd.read_csv("data/WorldQuant_Financial_Datasets.csv")
df_alphas = pd.read_csv("data/alpha50.csv")

# Initialize RL environment
env = AlphaTradingEnv(df_market_data, df_alphas)

# Train PPO model
ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=2e-4, n_steps=2048, batch_size=64, n_epochs=10)
ppo_model.learn(total_timesteps=200000)

# Save trained RL model
ppo_model.save("models/ppo_alpha_optimizer")
