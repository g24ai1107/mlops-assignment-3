import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Step 1: Load the original sklearn model
model = joblib.load("model.joblib")

# Step 2: Load data
X, y = fetch_california_housing(return_X_y=True)
y_pred_original = model.predict(X)
r2_original = r2_score(y, y_pred_original)

# Step 3: Manually quantize (float64 → float32)
weights = model.coef_.astype(np.float32)
bias = np.array([model.intercept_], dtype=np.float32)

# Step 4: Save quantized model as compressed numpy file
np.savez_compressed("quantized_model.npz", weights=weights, bias=bias)

# Step 5: Load and predict with quantized model
data = np.load("quantized_model.npz")
w = data["weights"]
b = data["bias"]

y_pred_quant = np.dot(X, w) + b
r2_quant = r2_score(y, y_pred_quant)

# Step 6: Compare sizes
original_size = os.path.getsize("model.joblib") / 1024
quant_size = os.path.getsize("quantized_model.npz") / 1024

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(f"Original Sklearn R² Score     : {r2_original:.4f}")
print(f"Quantized NumPy R² Score      : {r2_quant:.4f}")
print(f"Original Model Size           : {original_size:.2f} KB")
print(f"Quantized Model Size          : {quant_size:.2f} KB")

