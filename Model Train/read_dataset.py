import pandas as pd 
pd.set_option('display.max_columns', None)

df = pd.read_csv("india_retail_store.csv")
print("India retail Data\n")
print(df.head())

df = pd.read_csv("K_class_sales_clean.csv")
print("Kolkata Data\n")
print(df.head())

