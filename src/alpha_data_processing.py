import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Tokenizer
from concurrent.futures import ProcessPoolExecutor

class AlphaDataProcessor:
    """Optimized pipeline for financial data preprocessing and Alpha expression engineering."""
    
    def __init__(self, alpha_path, financial_path, feature_path):
        self.alpha_path = alpha_path
        self.financial_path = financial_path
        self.feature_path = feature_path
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def load_data(self):
        """Loads all datasets into memory efficiently."""
        self.df_alphas = pd.read_csv(self.alpha_path)
        self.df_financial = pd.read_csv(self.financial_path)
        self.df_features = pd.read_csv(self.feature_path)

    def extract_metadata(self):
        """Extracts Alpha metadata efficiently using parallel processing."""
        metadata_fields = ["Decay", "Delay", "Neutralization", "Truncation"]

        def parse_metadata(setting_str, key):
            match = re.search(rf"'{key}': '([^']*)'", str(setting_str))
            return match.group(1) if match else "Unknown"

        with ProcessPoolExecutor() as executor:
            for field in metadata_fields:
                self.df_alphas[field] = list(executor.map(lambda x: parse_metadata(x, field), self.df_alphas["settingdict"]))

    def normalize_data(self):
        """Scales financial and feature-engineered datasets efficiently."""
        scaler = StandardScaler()
        self.df_financial.iloc[:, 2:] = scaler.fit_transform(self.df_financial.iloc[:, 2:])
        self.df_features.iloc[:, 2:] = scaler.fit_transform(self.df_features.iloc[:, 2:])
        joblib.dump(scaler, "models/scaler.pkl")  # Save the scaler for reuse

    def tokenize_expressions(self):
        """Tokenizes Alpha expressions efficiently."""
        self.df_alphas["tokenized_formula"] = self.df_alphas["formula"].apply(
            lambda x: self.tokenizer.encode(str(x), add_special_tokens=True))

    def process(self):
        """Runs full data preprocessing pipeline."""
        self.load_data()
        self.extract_metadata()
        self.normalize_data()
        self.tokenize_expressions()
        return self.df_alphas, self.df_financial, self.df_features

# Execute preprocessing
processor = AlphaDataProcessor("data/alpha50.csv", "data/WorldQuant_Financial_Datasets.csv", "data/Expanded_WorldQuant_Feature_Engineered_Alphas.csv")
df_alphas, df_financial, df_features = processor.process()
