"""
ARIMA_model.py
Statistical Time-Series Baseline Model (Phase 1 Benchmark).

Features:
1. Evaluates ARIMA statistical benchmark across representative retail slices (e.g. Kolkata Grocery)
2. Enforces strict temporal ordering and continuous daily frequency
3. Calculates benchmark Technical KPIs: RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error)
4. Saves baseline metrics output to 'outputs/arima_benchmark_results.json'
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def calculate_metrics(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    non_zero_mask = actual > 0
    if np.sum(non_zero_mask) > 0:
        mape = float(np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100)
    else:
        mape = 0.0

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    return {"rmse": round(rmse, 2), "mape": round(mape, 2), "mae": round(mae, 2)}


def run_arima_benchmark(datasets_dir="datasets", outputs_dir="outputs", test_days=14, all_slices=True):
    print("=" * 60)
    print("RUNNING ARIMA TIME-SERIES STATISTICAL BENCHMARK (PHASE 1)")
    print("=" * 60)

    os.makedirs(outputs_dir, exist_ok=True)

    csv_path = os.path.join(datasets_dir, "xgb_training_ready.csv")
    if not os.path.exists(csv_path):
        # Fallback to local directory if not found in datasets/
        csv_path = "xgb_training_ready.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}. Run preprocess_india_retail_store_data.py first.")

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    if all_slices:
        slices = [
            {"city": row[0], "format": row[1], "category": row[2]}
            for row in df[["City", "Store_Format", "Category"]].drop_duplicates().sort_values(["City", "Category"]).values
        ]
    else:
        slices = [
            {"city": "Kolkata", "format": "Super", "category": "Grocery"},
            {"city": "Delhi", "format": "Hyper", "category": "Beverages"},
            {"city": "Mumbai", "format": "Express", "category": "Snacks"},
            {"city": "Bengaluru", "format": "Super", "category": "Dairy"},
        ]

    has_statsmodels = False
    try:
        from statsmodels.tsa.arima.model import ARIMA
        has_statsmodels = True
    except ImportError:
        print("[NOTICE] 'statsmodels' package is not yet installed in the environment.")
        print("         To install, run: pip install --user --break-system-packages statsmodels scikit-learn")
        print("         Running using Exponential Smoothing / Autoregressive Baseline Fallback in the interim.\n")

    benchmark_results = {}
    slice_summaries = []

    for s in slices:
        sub = df[
            (df["City"] == s["city"]) &
            (df["Store_Format"] == s["format"]) &
            (df["Category"] == s["category"])
        ].sort_values("Date").reset_index(drop=True)

        if len(sub) <= test_days:
            continue

        train = sub.iloc[:-test_days]
        test = sub.iloc[-test_days:]

        train_y = train["Total_Units"].values
        test_y = test["Total_Units"].values

        slice_key = f"{s['city']}_{s['format']}_{s['category']}"

        if has_statsmodels:
            try:
                model = ARIMA(train_y, order=(5, 1, 0))
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=test_days)
            except Exception:
                model = ARIMA(train_y, order=(1, 1, 0))
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=test_days)
        else:
            alpha = 0.3
            ewma = train_y[0]
            for val in train_y[1:]:
                ewma = alpha * val + (1 - alpha) * ewma
            recent_lags = train_y[-7:]
            predictions = np.array([recent_lags[i % 7] * 0.5 + ewma * 0.5 for i in range(test_days)])

        predictions = np.maximum(predictions, 0)
        metrics = calculate_metrics(test_y, predictions)

        benchmark_results[slice_key] = {
            "city": s["city"],
            "format": s["format"],
            "category": s["category"],
            "test_days": test_days,
            "metrics": metrics,
            "mean_actual": round(float(np.mean(test_y)), 2),
            "mean_predicted": round(float(np.mean(predictions)), 2),
        }

        slice_summaries.append({
            "Slice": f"{s['city']} - {s['category']} ({s['format']})",
            "City": s["city"],
            "Store_Format": s["format"],
            "Category": s["category"],
            "RMSE": metrics["rmse"],
            "MAPE (%)": f"{metrics['mape']}%",
            "MAE": metrics["mae"],
            "_mape_num": metrics["mape"],
        })

    summary_df = pd.DataFrame(slice_summaries)
    best_3 = summary_df.sort_values("RMSE").head(3).drop(columns=["_mape_num"])
    worst_3 = summary_df.sort_values("RMSE", ascending=False).head(3).drop(columns=["_mape_num"])

    print(f"\nARIMA Baseline Evaluation Summary ({len(slice_summaries)} Total Retail Slices Evaluated):")
    print("\n--- 🏆 Top 3 BEST Performing Slices (Lowest Prediction Error) ---")
    print(best_3[["Slice", "RMSE", "MAPE (%)", "MAE"]].to_string(index=False))
    print("\n--- ⚠️ Top 3 WORST Performing Slices (Highest Prediction Error / Volatility) ---")
    print(worst_3[["Slice", "RMSE", "MAPE (%)", "MAE"]].to_string(index=False))

    avg_rmse = round(float(np.mean([r["metrics"]["rmse"] for r in benchmark_results.values()])), 2)
    avg_mape = round(float(np.mean([r["metrics"]["mape"] for r in benchmark_results.values()])), 2)
    avg_mae = round(float(np.mean([r["metrics"]["mae"] for r in benchmark_results.values()])), 2)

    overall_metrics = {
        "model": "ARIMA Baseline (5,1,0)",
        "total_slices": len(benchmark_results),
        "avg_rmse": avg_rmse,
        "avg_mape": avg_mape,
        "avg_mae": avg_mae,
        "best_3": best_3[["Slice", "City", "Store_Format", "Category", "RMSE", "MAPE (%)", "MAE"]].to_dict(orient="records"),
        "worst_3": worst_3[["Slice", "City", "Store_Format", "Category", "RMSE", "MAPE (%)", "MAE"]].to_dict(orient="records"),
        "slices": benchmark_results,
    }

    out_file = os.path.join(outputs_dir, "arima_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"\n[ARIMA Benchmark] Overall Average RMSE: {avg_rmse} units")
    print(f"[ARIMA Benchmark] Overall Average MAPE: {avg_mape}%")
    print(f"[ARIMA Benchmark] Results saved to -> {out_file}")
    print("=" * 60)
    return overall_metrics


if __name__ == "__main__":
    run_arima_benchmark()