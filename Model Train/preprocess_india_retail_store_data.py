"""
preprocess_india_retail_store_data.py
Complete Data Engineering & Preprocessing Pipeline for Phase 1.

Key capabilities:
1. Data Cleaning & Anomaly Correction (Customer_Age median imputation, category cleanup)
2. Censored Demand (Stockouts) Detection & Imputation
3. Zero-Inflated Data Mitigation: Full date-grid expansion and Weekly Aggregation option
4. Contextual Enrichment:
   - Indian National Holidays & Major Retail Festivals (holidays lib + robust calendar)
   - Weather Signals (Avg_Temperature_C & Precipitation_mm via weather_enrichment)
   - Temporal Pay-Cycles (Is_Payday: 1st, 30th, 31st; Day_of_Week; Month)
5. Robust Lag & Rolling Feature Engineering (strictly grouped per City, Store_Format, Category)
6. Kolkata Store Outlet Lookalike Profiling (for Phase 2 digital maturity routing)

File Routing:
- All generated feature-engineered datasets are saved to the 'datasets/' subfolder.
"""

import os
import datetime
import numpy as np
import pandas as pd
from weather_enrichment import get_weather_dataframe

# Major Indian retail festivals for 2024 (creating major grocery/retail surges)
FESTIVALS_2024 = {
    # Makar Sankranti / Pongal
    "2024-01-14": "Makar Sankranti",
    "2024-01-15": "Pongal",
    # Republic Day
    "2024-01-26": "Republic Day",
    # Maha Shivratri
    "2024-03-08": "Maha Shivratri",
    # Holi
    "2024-03-24": "Holika Dahan",
    "2024-03-25": "Holi",
    # Eid al-Fitr
    "2024-04-11": "Eid al-Fitr",
    # Ram Navami
    "2024-04-17": "Ram Navami",
    # Bakrid / Eid al-Adha
    "2024-06-17": "Eid al-Adha",
    # Independence Day
    "2024-08-15": "Independence Day",
    # Raksha Bandhan
    "2024-08-19": "Raksha Bandhan",
    # Janmashtami
    "2024-08-26": "Janmashtami",
    # Ganesh Chaturthi
    "2024-09-07": "Ganesh Chaturthi",
    # Gandhi Jayanti
    "2024-10-02": "Gandhi Jayanti",
    # Navratri / Durga Puja (Massive FMCG & Grocery spike, especially Kolkata)
    "2024-10-09": "Durga Puja Sasthi",
    "2024-10-10": "Durga Puja Saptami",
    "2024-10-11": "Durga Puja Ashtami",
    "2024-10-12": "Dussehra / Vijaya Dashami",
    # Diwali & Dhanteras
    "2024-10-29": "Dhanteras",
    "2024-10-31": "Diwali",
    "2024-11-01": "Govardhan Puja",
    "2024-11-02": "Bhai Dooj",
    # Chhath Puja
    "2024-11-07": "Chhath Puja",
    # Guru Nanak Jayanti
    "2024-11-15": "Guru Nanak Jayanti",
    # Christmas
    "2024-12-25": "Christmas",
    # New Year's Eve
    "2024-12-31": "New Year's Eve",
}


def get_indian_holiday_flags(dates):
    """
    Returns holiday flag (1 or 0) for each date in series.
    Uses 'holidays' library if installed, supplemented by FESTIVALS_2024.
    """
    holiday_dates = set(FESTIVALS_2024.keys())

    try:
        import holidays
        in_holidays = holidays.India(years=[2023, 2024, 2025])
        for dt_obj in in_holidays:
            holiday_dates.add(dt_obj.strftime("%Y-%m-%d"))
    except ImportError:
        pass

    return [1 if str(d) in holiday_dates else 0 for d in dates]


def impute_censored_demand(daily_df):
    """
    Censored Demand Imputation:
    When Stock_On_Hand is at or below Reorder_Level, observed units sold are constrained
    by stockout conditions. We estimate the unconstrained latent demand using the expected
    sales under normal inventory conditions for that segment and day of week.
    """
    print("[Preprocessing] Running Censored Demand (Stockout) Imputation...")
    non_censored = daily_df[daily_df["Stock_On_Hand"] > daily_df["Reorder_Level"]]
    baseline_demand = non_censored.groupby(
        ["City", "Store_Format", "Category", "Day_of_Week"]
    )["Total_Units"].mean().reset_index(name="Expected_Uncensored_Units")

    merged = pd.merge(
        daily_df,
        baseline_demand,
        on=["City", "Store_Format", "Category", "Day_of_Week"],
        how="left"
    )

    overall_cat_mean = daily_df.groupby("Category")["Total_Units"].transform("mean")
    merged["Expected_Uncensored_Units"] = merged["Expected_Uncensored_Units"].fillna(overall_cat_mean)

    is_censored = merged["Stock_On_Hand"] <= merged["Reorder_Level"]
    merged["Is_Stockout_Censored"] = is_censored.astype(int)

    imputed_units = np.where(
        is_censored & (merged["Expected_Uncensored_Units"] > merged["Total_Units"]),
        np.round(merged["Expected_Uncensored_Units"]),
        merged["Total_Units"]
    )
    merged["Imputed_Demand"] = imputed_units
    censored_count = merged["Is_Stockout_Censored"].sum()
    print(f"[Preprocessing] Identified {censored_count} censored records ({censored_count/len(merged):.1%}) - latent demand recovered.")
    return merged


def build_full_daily_time_series(india_df):
    """
    Creates continuous Cartesian product of Date x City x Store_Format x Category
    to resolve missing dates / zero-inflated retail records.
    """
    india_df["Invoice_Date"] = pd.to_datetime(india_df["Invoice_Date"])
    india_df["Date"] = india_df["Invoice_Date"].dt.date
    india_df["Customer_Age"] = india_df["Customer_Age"].fillna(india_df["Customer_Age"].median())

    raw_agg = india_df.groupby(["Date", "City", "Store_Format", "Category"]).agg(
        Total_Units=("Units", "sum"),
        Avg_Selling_Price=("Selling_Price", "mean"),
        Stock_On_Hand=("Stock_On_Hand", "mean"),
        Reorder_Level=("Reorder_Level", "max"),
    ).reset_index()

    unique_dates = pd.date_range(start="2024-01-01", end="2024-12-30", freq="D").date
    unique_cities = india_df["City"].unique()
    unique_formats = india_df["Store_Format"].unique()
    unique_cats = india_df["Category"].unique()

    grid_index = pd.MultiIndex.from_product(
        [unique_dates, unique_cities, unique_formats, unique_cats],
        names=["Date", "City", "Store_Format", "Category"]
    )
    full_grid = pd.DataFrame(index=grid_index).reset_index()

    merged = pd.merge(full_grid, raw_agg, on=["Date", "City", "Store_Format", "Category"], how="left")
    merged["Total_Units"] = merged["Total_Units"].fillna(0)

    merged = merged.sort_values(by=["City", "Store_Format", "Category", "Date"])
    merged["Avg_Selling_Price"] = merged.groupby(["City", "Store_Format", "Category"])["Avg_Selling_Price"].ffill().bfill()
    merged["Stock_On_Hand"] = merged.groupby(["City", "Store_Format", "Category"])["Stock_On_Hand"].ffill().bfill()
    merged["Reorder_Level"] = merged.groupby(["City", "Store_Format", "Category"])["Reorder_Level"].ffill().bfill()

    return merged


def find_file(filename, search_dirs=(".", "datasets")):
    """Helper to locate input dataset files in search directories."""
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(search_dirs[0], filename)


def preprocess_all(data_dir=".", datasets_dir="datasets"):
    print("=" * 60)
    print("STARTING DATA PREPROCESSING & ENRICHMENT PIPELINE (PHASE 1)")
    print("=" * 60)

    os.makedirs(datasets_dir, exist_ok=True)

    india_csv = find_file("india_retail_store.csv", [datasets_dir, data_dir])
    kolkata_csv = find_file("kolkata_data.csv", [datasets_dir, data_dir])

    if not os.path.exists(india_csv):
        raise FileNotFoundError(f"Missing {india_csv}")

    print(f"[Preprocessing] Loading input dataset: {india_csv}...")
    india_df = pd.read_csv(india_csv)

    # Step 1: Build full time-series grid
    daily_df = build_full_daily_time_series(india_df)

    # Step 2: Temporal Features
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df["Day_of_Week"] = daily_df["Date"].dt.dayofweek
    daily_df["Month"] = daily_df["Date"].dt.month
    daily_df["Day"] = daily_df["Date"].dt.day
    daily_df["Is_Weekend"] = daily_df["Day_of_Week"].apply(lambda x: 1 if x in [5, 6] else 0)
    daily_df["Is_Payday"] = daily_df["Day"].apply(lambda x: 1 if x in [1, 30, 31] else 0)

    # Step 3: Contextual Holiday and Festival Flags
    date_strs = daily_df["Date"].dt.strftime("%Y-%m-%d").values
    daily_df["Is_Holiday"] = get_indian_holiday_flags(date_strs)

    # Step 4: Weather Enrichment
    cities = daily_df["City"].unique().tolist()
    weather_df = get_weather_dataframe(cities, start_date="2024-01-01", end_date="2024-12-30")
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    daily_df = pd.merge(daily_df, weather_df, on=["Date", "City"], how="left")

    # Step 5: Censored Demand (Stockout) Imputation
    daily_df = impute_censored_demand(daily_df)

    # Step 6: Mathematically Correct Lags & Rolling Features per Time Series
    print("[Preprocessing] Engineering Lags and Rolling Windows (per City, Store_Format, Category)...")
    daily_df = daily_df.sort_values(by=["City", "Store_Format", "Category", "Date"]).reset_index(drop=True)
    group_cols = ["City", "Store_Format", "Category"]

    daily_df["Units_Lag_7d"] = daily_df.groupby(group_cols)["Total_Units"].shift(7)
    daily_df["Units_Lag_14d"] = daily_df.groupby(group_cols)["Total_Units"].shift(14)
    daily_df["Units_Lag_28d"] = daily_df.groupby(group_cols)["Total_Units"].shift(28)

    daily_df["Units_Rolling_Mean_7d"] = daily_df.groupby(group_cols)["Total_Units"].transform(lambda x: x.shift(1).rolling(7).mean())
    daily_df["Units_Rolling_Mean_14d"] = daily_df.groupby(group_cols)["Total_Units"].transform(lambda x: x.shift(1).rolling(14).mean())
    daily_df["Units_Rolling_Std_7d"] = daily_df.groupby(group_cols)["Total_Units"].transform(lambda x: x.shift(1).rolling(7).std()).fillna(0)

    clean_daily = daily_df.dropna(subset=["Units_Lag_28d"]).copy()

    # Save to datasets/xgb_training_ready.csv
    out_daily_path = os.path.join(datasets_dir, "xgb_training_ready.csv")
    clean_daily.to_csv(out_daily_path, index=False)
    print(f"[Preprocessing] Saved engineered dataset -> {out_daily_path} ({len(clean_daily)} rows, {clean_daily.shape[1]} columns)")

    # Step 7: Weekly Aggregation Dataset
    print("[Preprocessing] Generating Weekly Aggregated Dataset for zero-inflation smoothing...")
    clean_daily["Year"] = clean_daily["Date"].dt.isocalendar().year
    clean_daily["Week"] = clean_daily["Date"].dt.isocalendar().week

    weekly_df = clean_daily.groupby(["Year", "Week", "City", "Store_Format", "Category"]).agg(
        Total_Units=("Total_Units", "sum"),
        Imputed_Demand=("Imputed_Demand", "sum"),
        Avg_Selling_Price=("Avg_Selling_Price", "mean"),
        Stock_On_Hand=("Stock_On_Hand", "mean"),
        Reorder_Level=("Reorder_Level", "max"),
        Avg_Temperature_C=("Avg_Temperature_C", "mean"),
        Total_Precipitation_mm=("Precipitation_mm", "sum"),
        Holiday_Days_Count=("Is_Holiday", "sum"),
        Payday_Days_Count=("Is_Payday", "sum"),
    ).reset_index()

    weekly_group = ["City", "Store_Format", "Category"]
    weekly_df = weekly_df.sort_values(by=weekly_group + ["Year", "Week"]).reset_index(drop=True)
    weekly_df["Units_Lag_1w"] = weekly_df.groupby(weekly_group)["Total_Units"].shift(1)
    weekly_df["Units_Lag_2w"] = weekly_df.groupby(weekly_group)["Total_Units"].shift(2)
    weekly_df["Units_Lag_4w"] = weekly_df.groupby(weekly_group)["Total_Units"].shift(4)
    weekly_df["Units_Rolling_Mean_4w"] = weekly_df.groupby(weekly_group)["Total_Units"].transform(lambda x: x.shift(1).rolling(4).mean())
    clean_weekly = weekly_df.dropna(subset=["Units_Lag_4w"]).copy()

    # Save to datasets/weekly_demand_training_ready.csv
    out_weekly_path = os.path.join(datasets_dir, "weekly_demand_training_ready.csv")
    clean_weekly.to_csv(out_weekly_path, index=False)
    print(f"[Preprocessing] Saved weekly aggregated dataset -> {out_weekly_path} ({len(clean_weekly)} rows)")

    # Step 8: Kolkata Shop Sales Profiling
    if os.path.exists(kolkata_csv):
        print(f"[Preprocessing] Processing Kolkata Shop Sales -> {kolkata_csv}...")
        kolkata_df = pd.read_csv(kolkata_csv)
        kolkata_df["Item_Fat_Content"] = kolkata_df["Item_Fat_Content"].replace(
            {"low fat": "Low Fat", "LF": "Low Fat", "reg": "Regular"}
        )
        outlet_profiles = kolkata_df.groupby("Outlet_Identifier").agg(
            Avg_Daily_Sales=("Item_Outlet_Sales", "mean"),
            Total_Profit=("Profit", "sum"),
            Outlet_Size=("Outlet_Size", "first"),
            Location_Tier=("Outlet_Location_Type", "first"),
            Total_SKUs=("Item_Identifier", "nunique")
        ).reset_index()
        # Save to datasets/outlet_lookalike_profiles.csv
        out_profile_path = os.path.join(datasets_dir, "outlet_lookalike_profiles.csv")
        outlet_profiles.to_csv(out_profile_path, index=False)
        print(f"[Preprocessing] Saved outlet lookalike profiles dataset -> {out_profile_path}")

    print("=" * 60)
    print("DATA ENGINEERING & PREPROCESSING COMPLETE!")
    print(f"All generated datasets saved to folder: {os.path.abspath(datasets_dir)}")
    print("=" * 60)
    return clean_daily, clean_weekly


if __name__ == "__main__":
    preprocess_all()
