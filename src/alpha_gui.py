import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from alpha_pipeline import AlphaPipeline

# Initialize pipeline with live market data support
pipeline = AlphaPipeline(
    transformer_model_path="models/optimized_alpha_transformer.pth",
    rl_model_path="models/ppo_alpha_optimizer.zip",
    market_data_path="data/WorldQuant_Financial_Datasets.csv",
    alpha_data_path="data/alpha50.csv",
    use_live_data=True
)

class AlphaGUI:
    """GUI-based tool for real-time Alpha generation, optimization, and backtesting."""

    def __init__(self, root):
        self.root = root
        self.root.title("Alpha Generator & Live Trading")
        self.root.geometry("1000x700")

        # Alpha Expression Entry
        self.label_alpha = ttk.Label(root, text="Generated Alpha Expression:")
        self.label_alpha.pack()
        self.alpha_var = tk.StringVar()
        self.entry_alpha = ttk.Entry(root, textvariable=self.alpha_var, width=80)
        self.entry_alpha.pack(pady=5)

        # Generate Button
        self.btn_generate = ttk.Button(root, text="Generate Alpha", command=self.generate_alpha)
        self.btn_generate.pack(pady=5)

        # Optimize Button
        self.btn_optimize = ttk.Button(root, text="Optimize Alpha (RL)", command=self.optimize_alpha)
        self.btn_optimize.pack(pady=5)

        # Backtest Button
        self.btn_backtest = ttk.Button(root, text="Backtest Alpha", command=self.run_backtest_thread)
        self.btn_backtest.pack(pady=5)

        # Execute Live Alpha Button
        self.btn_live_execute = ttk.Button(root, text="Execute Live Alpha", command=self.run_live_execution_thread)
        self.btn_live_execute.pack(pady=5)

        # Performance Metrics Display
        self.metrics_frame = ttk.LabelFrame(root, text="Performance Metrics")
        self.metrics_frame.pack(pady=10, fill="both", expand=True)
        self.metrics_text = tk.Text(self.metrics_frame, height=10, width=80)
        self.metrics_text.pack()

    def generate_alpha(self):
        """Generates an Alpha formula using the Transformer model."""
        generated_alpha = pipeline.generate_alpha()
        self.alpha_var.set(generated_alpha)

    def optimize_alpha(self):
        """Optimizes the generated Alpha using reinforcement learning."""
        optimized_alpha = pipeline.optimize_alpha(self.alpha_var.get())
        self.alpha_var.set(optimized_alpha)

    def run_backtest_thread(self):
        """Runs backtest in a separate thread to prevent UI freeze."""
        threading.Thread(target=self.backtest_alpha, daemon=True).start()

    def backtest_alpha(self):
        """Backtests the generated Alpha expression and updates results dynamically."""
        alpha_expr = self.alpha_var.get()
        if not alpha_expr:
            messagebox.showerror("Error", "No Alpha expression provided!")
            return
        
        backtest_results = pipeline.backtest_alpha(alpha_expr)

        self.metrics_text.delete(1.0, tk.END)
        for key, value in backtest_results.items():
            self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")

        self.plot_results(alpha_expr)

    def run_live_execution_thread(self):
        """Runs live Alpha execution in a separate thread."""
        threading.Thread(target=self.execute_live_alpha, daemon=True).start()

    def execute_live_alpha(self):
        """Executes the Alpha strategy using real-time market data."""
        alpha_expr = self.alpha_var.get()
        if not alpha_expr:
            messagebox.showerror("Error", "No Alpha expression provided!")
            return

        live_results = pipeline.backtest_alpha(alpha_expr)

        self.metrics_text.delete(1.0, tk.END)
        for key, value in live_results.items():
            self.metrics_text.insert(tk.END, f"{key}: {value:.4f}\n")

        self.plot_results(alpha_expr)

    def plot_results(self, alpha_expr):
        """Plots cumulative returns of the Alpha strategy."""
        df_alpha = pipeline.backtest_alpha(alpha_expr)

        if df_alpha is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(df_alpha["strategy_returns"].cumsum(), label="Strategy Returns")
            plt.legend()
            plt.title("Cumulative Returns (Live)")
            plt.xlabel("Time")
            plt.ylabel("Returns")

            # Embed plot in Tkinter
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
            canvas.get_tk_widget().pack()
            canvas.draw()

# Run GUI Application
root = tk.Tk()
app = AlphaGUI(root)
root.mainloop()
