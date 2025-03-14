import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class AlphaTransformer:
    """Optimized GPT-2 Transformer for Alpha formula generation."""

    def __init__(self, model_name="gpt2", lr=2e-5, batch_size=16, epochs=10):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, dataset):
        """Trains GPT-2 Transformer on tokenized Alpha expressions."""
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataset:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

        torch.save(self.model.state_dict(), "models/optimized_alpha_transformer.pth")

    def generate_alpha(self, seed_text="rank(close)", temperature=0.7, top_p=0.9):
        """Generates an Alpha formula with controlled randomness."""
        input_ids = self.tokenizer.encode(seed_text, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=temperature, top_p=top_p)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
transformer = AlphaTransformer()
generated_alpha = transformer.generate_alpha()
print(f"Generated Alpha: {generated_alpha}")
