import joblib
import numpy as np
import os
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train original sklearn model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
original_r2 = r2_score(y_test, y_pred_sklearn)

# Extract weights and bias
weights = model.coef_
bias = model.intercept_

# Save unquantized parameters
params = {"weights": weights, "bias": bias}
joblib.dump(params, "unquant_params.joblib")

# -----------------------------
# Quantization (Separate for weights and bias)
# -----------------------------
w_min, w_max = weights.min(), weights.max()
w_scale = (w_max - w_min) / 255
w_zero_point = np.round(-w_min / w_scale)

q_weights = np.round(weights / w_scale + w_zero_point).astype(np.uint8)

b_min, b_max = bias, bias
b_scale = (b_max - b_min + 1e-6) / 255
b_zero_point = np.round(-b_min / b_scale)
q_bias = np.round(bias / b_scale + b_zero_point).astype(np.uint8)

# Save quantized parameters
quant_params = {
    "weights": q_weights,
    "bias": q_bias,
    "w_scale": w_scale,
    "w_zero_point": w_zero_point,
    "b_scale": b_scale,
    "b_zero_point": b_zero_point
}
joblib.dump(quant_params, "quant_params.joblib")

# -----------------------------
# Dequantization
# -----------------------------
dq_weights = w_scale * (q_weights.astype(np.float32) - w_zero_point)
dq_bias = b_scale * (q_bias.astype(np.float32) - b_zero_point)

# -----------------------------
# PyTorch Model
# -----------------------------
class LinearModel(torch.nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
        self.linear.weight = torch.nn.Parameter(torch.tensor(weights.reshape(1, -1), dtype=torch.float32))
        self.linear.bias = torch.nn.Parameter(torch.tensor([bias], dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)

# Inference using PyTorch model
model = LinearModel(dq_weights, dq_bias)
with torch.no_grad():
    y_pred_torch = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
quantized_r2 = r2_score(y_test, y_pred_torch)

# -----------------------------
# Report
# -----------------------------
print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(f"Original Sklearn R² Score     : {original_r2:.4f}")
print(f"Quantized PyTorch R² Score    : {quantized_r2:.4f}")

# Get file sizes
def get_file_size(file):
    return os.path.getsize(file) / 1024  # KB

original_size = get_file_size("unquant_params.joblib")
quant_size = get_file_size("quant_params.joblib")

print(f"Original Model Size           : {original_size:.2f} KB")
print(f"Quantized Model Size          : {quant_size:.2f} KB")

