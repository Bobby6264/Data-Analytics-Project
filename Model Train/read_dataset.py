import os
import pandas as pd

pd.set_option('display.max_columns', None)

datasets_dir = "datasets" if os.path.exists("datasets") else "."
india_path = os.path.join(datasets_dir, "india_retail_store.csv")
kolkata_path = os.path.join(datasets_dir, "kolkata_data.csv")

output_lines = []

output_lines.append("India retail Data\n")
df_india = pd.read_csv(india_path)
output_lines.append(df_india.head().to_string())

output_lines.append("\nKolkata Data\n")
df_kolkata = pd.read_csv(kolkata_path)
output_lines.append(df_kolkata.head().to_string())

output_text = "\n".join(output_lines)
print(output_text)

# Save dataset summary directly to datasets/dataset.txt
out_txt = os.path.join(datasets_dir, "dataset.txt")
with open(out_txt, "w") as f:
    f.write(output_text)
print(f"\n[Info] Saved summary to -> {out_txt}")
