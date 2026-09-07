import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Load the prepared data
df = pd.read_csv("xgb_training_ready.csv")

# Filter for a single City and Category for the ARIMA baseline (e.g., Kolkata Grocery)
baseline_df = df[(df['City'] == 'Kolkata') & (df['Category'] == 'Grocery')]
train = baseline_df.iloc[:-14] # Train on all but the last 14 days
test = baseline_df.iloc[-14:]  # Test on the last 14 days

# Train ARIMA baseline
model = ARIMA(train['Total_Units'], order=(5, 1, 0))
model_fit = model.fit()

# Calculate Benchmark KPIs
predictions = model_fit.forecast(steps=len(test))
rmse = root_mean_squared_error(test['Total_Units'], predictions)
mape = mean_absolute_percentage_error(test['Total_Units'], predictions)

print(f"ARIMA Baseline RMSE: {rmse:.2f}")
print(f"ARIMA Baseline MAPE: {mape:.2f}")