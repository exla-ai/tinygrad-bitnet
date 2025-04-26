from safetensors import safe_open
import os

# --- Adjust this path ---
# Usually downloaded by tinygrad's fetch to ~/.cache/tinygrad/downloads/
model_dir = "/Users/viraat/Documents/projects/exla/bitnet-accelerator/bitnet_model_artifacts_b1.58-2B-4T"
safetensors_path = os.path.join(model_dir, "model.safetensors")
# Or provide the direct path if you downloaded it elsewhere
# safetensors_path = "/path/to/your/model.safetensors"
# -----------------------

if not os.path.exists(safetensors_path):
    print(f"Error: File not found at {safetensors_path}")
    print("Please check the path or run the example script first to download the model.")
else:
    print(f"Analyzing tensors in: {safetensors_path}")
    tensors = {}
    # Use safe_open for lazy loading (doesn't load actual tensor data)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
         # Iterate through all tensor names (keys)
         for key in f.keys():
             print(key)
             # Optionally, you could load a specific tensor's shape/dtype
             # tensor_info = f.get_tensor(key)
             # print(f"  Shape: {tensor_info.shape}, Dtype: {tensor_info.dtype}")

print("\nAnalysis complete.")

