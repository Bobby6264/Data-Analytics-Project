import kagglehub

# Download latest version
path = kagglehub.dataset_download("sushantchougule/kolkata-shops-sales")

print("Path to dataset files:", path)