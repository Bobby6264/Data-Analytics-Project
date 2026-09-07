import pandas as pd
import numpy as np 
import holidays

india_df = pd.read_csv("india_retail_store.csv")

india_df['Invoice_Date'] = pd.to_datetime(india_df['Invoice_Date'])
india_df['Date'] = india_df['Invoice_Date'].dt.date

india_df['Customer_Age'] = india_df['Customer_Age'].fillna(india_df['Customer_Age'].median())

daily_demand = india_df.groupby(['Date', 'City', 'Store_Format', 'Category']).agg(
    Total_Units=('Units', 'sum'),
    Avg_Selling_Price=('Selling_Price', 'mean'),
    Stock_On_Hand=('Stock_On_Hand', 'mean'), # Average daily stock
    Reorder_Level=('Reorder_Level', 'max')
).reset_index()

in_holidays = holidays.India(years=[2023,2024,2025])
daily_demand['Is_Holiday'] = daily_demand['Date'].apply(lambda x: 1 if x in in_holidays else 0)
daily_demand['Date'] = pd.to_datetime(daily_demand['Date'])
daily_demand['Day_of_Week'] = daily_demand['Date'].dt.dayofweek
daily_demand['Month'] = daily_demand['Date'].dt.month
daily_demand['Is_Payday'] = daily_demand['Date'].dt.day.apply(lambda x: 1 if x in [1, 30, 31] else 0) # Temporal pay-cycle insight

# Sort sequentially for time-series integrity
daily_demand = daily_demand.sort_values(by=['City', 'Category', 'Date'])

# Create 7-day and 14-day rolling demand lags
daily_demand['Units_Lag_7d'] = daily_demand.groupby(['City', 'Category'])['Total_Units'].shift(7)
daily_demand['Units_Lag_14d'] = daily_demand.groupby(['City', 'Category'])['Total_Units'].shift(14)

# Drop NaN values created by the shift operation
daily_demand = daily_demand.dropna()

# Save the engineered features for model training
daily_demand.to_csv("xgb_training_ready.csv", index=False)


kolkata_df = pd.read_csv("kolkata_data.csv")

# Standardize categorical anomalies (e.g., 'Low Fat', 'low fat', 'LF' are often mixed in this dataset)
kolkata_df['Item_Fat_Content'] = kolkata_df['Item_Fat_Content'].replace(
    {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}
)

# Create an Outlet Profile matrix
outlet_profiles = kolkata_df.groupby('Outlet_Identifier').agg(
    Avg_Daily_Sales=('Item_Outlet_Sales', 'mean'),
    Total_Profit=('Profit', 'sum'),
    Outlet_Size=('Outlet_Size', 'first'),
    Location_Tier=('Outlet_Location_Type', 'first')
).reset_index()

# Save for the backend API to use when routing "No Data" shops
outlet_profiles.to_csv("outlet_lookalike_profiles.csv", index=False)


