import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from stable_baselines3 import PPO
from backtesting_engine import AdvancedAlphaBacktester
from reinforcement_learning import AlphaTradingEnv
from live_market_data import LiveMarketData

class AlphaPipeline:
    """Integrated pipeline for Alpha generation, optimization, and real-time backtesting."""

    def __init__(self, transformer_model_path, rl_model_path, market_data_path, alpha_data_path, use_live_data=False):
        # Load Transformer model for Alpha generation
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.transformer = GPT2LMHeadModel.from_pretrained("gpt2")
        self.transformer.load_state_dict(torch.load(transformer_model_path, map_location="cpu"))
        self.transformer.eval()

        # Load PPO reinforcement learning model
        self.df_market_data = pd.read_csv(market_data_path)
        self.df_alphas = pd.read_csv(alpha_data_path)
        self.env = AlphaTradingEnv(self.df_market_data, self.df_alphas)
        self.rl_model = PPO.load(rl_model_path)

        # Real-time data integration
        self.use_live_data = use_live_data
        self.live_data_source = LiveMarketData()

    def generate_alpha(self, seed_text="rank(close)", temperature=0.7, top_p=0.9):
        """Generates an Alpha expression using the Transformer model."""
        input_ids = self.tokenizer.encode(seed_text, return_tensors="pt")
        output = self.transformer.generate(input_ids, max_length=50, temperature=temperature, top_p=top_p)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def optimize_alpha(self, alpha_expr):
        """Uses PPO reinforcement learning to fine-tune Alpha expressions."""
        obs = self.env.reset()
        action, _states = self.rl_model.predict(obs)
        optimized_alpha = self.df_alphas.iloc[action]["formula"]
        return optimized_alpha

    def backtest_alpha(self, alpha_expr):
        """Runs backtesting on the given Alpha expression."""
        backtester = AdvancedAlphaBacktester(self.df_market_data, pd.DataFrame({"formula": [alpha_expr]}), self.live_data_source if self.use_live_data else None)
        return backtester.execute_live_alpha(alpha_expr) if self.use_live_data else backtester.backtest_alpha(alpha_expr)

# Initialize full pipeline
pipeline = AlphaPipeline(
    transformer_model_path="models/optimized_alpha_transformer.pth",
    rl_model_path="models/ppo_alpha_optimizer.zip",
    market_data_path="data/WorldQuant_Financial_Datasets.csv",
    alpha_data_path="data/alpha50.csv",
    use_live_data=True
)

# Generate, optimize, and backtest an Alpha in real-time
generated_alpha = pipeline.generate_alpha()
optimized_alpha = pipeline.optimize_alpha(generated_alpha)
backtest_results = pipeline.backtest_alpha(optimized_alpha)

# Print results
print(f"Generated Alpha: {generated_alpha}")
print(f"Optimized Alpha: {optimized_alpha}")
print(f"Backtest Results: {backtest_results}")
