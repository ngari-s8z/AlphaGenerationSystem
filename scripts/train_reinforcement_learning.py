from stable_baselines3 import PPO
from src.reinforcement_learning import AlphaTradingEnv
import pandas as pd

# Load datasets
df_market_data = pd.read_csv("data/WorldQuant_Financial_Datasets.csv")
df_alphas = pd.read_csv("data/alpha50.csv")

# Initialize environment
env = AlphaTradingEnv(df_market_data, df_alphas)

# Train PPO model
ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=2e-4, n_steps=2048, batch_size=64, n_epochs=10)
ppo_model.learn(total_timesteps=200000)

# Save trained RL model
ppo_model.save("models/ppo_alpha_optimizer")
