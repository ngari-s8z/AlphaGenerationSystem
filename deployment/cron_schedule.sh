#!/bin/bash

# Run Transformer model training weekly
echo "0 3 * * 1 cd /path/to/AlphaGenerationSystem && python scripts/train_transformer.py" | crontab -

# Run PPO reinforcement learning daily at 2 AM
echo "0 2 * * * cd /path/to/AlphaGenerationSystem && python scripts/train_reinforcement_learning.py" | crontab -

# Run backtesting every 6 hours
echo "0 */6 * * * cd /path/to/AlphaGenerationSystem && python scripts/run_backtest.py" | crontab -
