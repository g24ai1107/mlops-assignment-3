import joblib
import numpy as np
import torch
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Step 1: Load original model
model = joblib.load("model.joblib")

# Step 2: Load data
X, y = fetch_california_housing(return_X_y=True)
y_pred = model.predict(X)
r2_original = r2_score(y, y_pred)

# Step 3: Quantize weights to float32
weights = model.coef_.astype(np.float32)
bias = np.array([model.intercept_], dtype=np.float32)

# Step 4: Define minimal PyTorch model
class TinyLinear(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create model
torch_model = TinyLinear(X.shape[1])

# Set weights manually
with torch.no_grad():
    torch_model.linear.weight = torch.nn.Parameter(torch.tensor(weights.reshape(1, -1)))
    torch_model.linear.bias = torch.nn.Parameter(torch.tensor(bias))

# Step 5: Save quantized model
quant_model_path = "quantized_model.pt"
torch.save(torch_model.state_dict(), quant_model_path)

# Step 6: Evaluate quantized model
X_tensor = torch.tensor(X.astype(np.float32))
y_tensor = torch.tensor(y.reshape(-1, 1).astype(np.float32))

with torch.no_grad():
    y_pred_quant = torch_model(X_tensor).numpy()

r2_quant = r2_score(y, y_pred_quant)

# Step 7: Compare sizes
original_size = os.path.getsize("model.joblib") / 1024  # KB
quant_size = os.path.getsize(quant_model_path) / 1024  # KB

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(f"Original Sklearn R² Score     : {r2_original:.4f}")
print(f"Quantized PyTorch R² Score    : {r2_quant:.4f}")
print(f"Original Model Size           : {original_size:.2f} KB")
print(f"Quantized Model Size          : {quant_size:.2f} KB")

