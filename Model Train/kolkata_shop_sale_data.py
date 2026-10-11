import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sushantchougule/kolkata-shops-sales")
print("Path to dataset files:", path)

# Target directory for datasets
datasets_dir = "datasets"
os.makedirs(datasets_dir, exist_ok=True)

# Copy downloaded csv directly into datasets/ folder
target_path = os.path.join(datasets_dir, "kolkata_data.csv")
for f in os.listdir(path):
    if f.endswith(".csv"):
        src = os.path.join(path, f)
        shutil.copy2(src, target_path)
        print(f"Saved dataset directly to -> {target_path}")
        break