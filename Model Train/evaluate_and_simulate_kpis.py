"""
evaluate_and_simulate_kpis.py
Business & Technical KPI Evaluation and Inventory Optimization Simulator.

Directly satisfies Phase 1 deliverables from PDF Slide 2 & 7:
1. Technical Benchmarks: RMSE & MAPE comparison (ARIMA vs. XGBoost vs. LSTM)
2. Business KPI Simulation:
   - Reduced Holding Costs (quantifying capital tied up in excess stock)
   - Lowered Spoilage / Shrinkage (for perishable goods like Dairy, Fruits, Vegetables)
   - Prevented Stockouts (recovering lost sales vs. static human intuition / fixed re-order points)
3. Outputs are saved to 'outputs/'.
"""

import json
import os
import sys
import numpy as np
import pandas as pd


def load_model_benchmarks(outputs_dir="outputs"):
    """
    Loads saved results from ARIMA, XGBoost, and LSTM runs.
    """
    benchmarks = {}

    arima_path = os.path.join(outputs_dir, "arima_benchmark_results.json")
    if not os.path.exists(arima_path):
        arima_path = "arima_benchmark_results.json"
    if os.path.exists(arima_path):
        with open(arima_path, "r") as f:
            benchmarks["ARIMA"] = json.load(f)

    xgb_path = os.path.join(outputs_dir, "xgboost_evaluation_results.json")
    if not os.path.exists(xgb_path):
        xgb_path = "xgboost_evaluation_results.json"
    if os.path.exists(xgb_path):
        with open(xgb_path, "r") as f:
            benchmarks["XGBoost"] = json.load(f)

    lstm_path = os.path.join(outputs_dir, "lstm_benchmark_results.json")
    if not os.path.exists(lstm_path):
        lstm_path = "lstm_benchmark_results.json"
    if os.path.exists(lstm_path):
        with open(lstm_path, "r") as f:
            benchmarks["LSTM"] = json.load(f)

    return benchmarks


def run_inventory_business_simulation(datasets_dir="datasets", outputs_dir="outputs"):
    print("=" * 60)
    print("RUNNING BUSINESS KPI & INVENTORY OPTIMIZATION SIMULATION")
    print("=" * 60)

    os.makedirs(outputs_dir, exist_ok=True)

    # Find predictions csv
    predictions_csv = os.path.join(outputs_dir, "xgboost_predictions.csv")
    if not os.path.exists(predictions_csv):
        predictions_csv = os.path.join(datasets_dir, "xgboost_predictions.csv")
        if not os.path.exists(predictions_csv):
            predictions_csv = "xgboost_predictions.csv"
            if not os.path.exists(predictions_csv):
                raise FileNotFoundError(f"Missing {predictions_csv}. Run xgboost_model.py first.")

    pred_df = pd.read_csv(predictions_csv)
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])

    perishable_categories = {"Dairy", "Fruits", "Vegetables"}
    annual_holding_cost_rate = 0.18
    daily_holding_rate = annual_holding_cost_rate / 365.0
    spoilage_rate_perishables = 0.05

    unique_series = pred_df.groupby(["City", "Store_Format", "Category"])

    static_stockout_events = 0
    ml_stockout_events = 0

    static_holding_costs = 0.0
    ml_holding_costs = 0.0

    static_spoilage_units = 0.0
    ml_spoilage_units = 0.0

    for (city, store_format, category), group in unique_series:
        group = group.sort_values("Date").reset_index(drop=True)
        is_perishable = category in perishable_categories
        unit_cost = group["Avg_Selling_Price"].mean() * 0.8

        static_inv = float(group["Stock_On_Hand"].iloc[0])
        ml_inv = float(group["Stock_On_Hand"].iloc[0])
        fixed_reorder_level = float(group["Reorder_Level"].mean())

        for idx, row in group.iterrows():
            actual_demand = float(row["Total_Units"])
            ml_predicted = float(row["Predicted_Units"])

            if static_inv <= fixed_reorder_level:
                static_order = fixed_reorder_level * 1.5
            else:
                static_order = 0.0

            if static_inv < actual_demand:
                static_stockout_events += 1
                static_sales = static_inv
                static_inv = 0.0
            else:
                static_sales = actual_demand
                static_inv -= actual_demand

            static_inv += static_order
            daily_static_hold = static_inv * unit_cost * daily_holding_rate
            static_holding_costs += daily_static_hold

            if is_perishable and static_inv > (actual_demand * 2):
                static_spoilage_units += (static_inv - actual_demand * 2) * spoilage_rate_perishables

            safety_stock = np.sqrt(ml_predicted) * 1.65
            dynamic_target = ml_predicted + safety_stock

            if ml_inv < dynamic_target:
                ml_order = max(0.0, dynamic_target - ml_inv)
            else:
                ml_order = 0.0

            if ml_inv < actual_demand:
                ml_stockout_events += 1
                ml_sales = ml_inv
                ml_inv = 0.0
            else:
                ml_sales = actual_demand
                ml_inv -= actual_demand

            ml_inv += ml_order
            daily_ml_hold = ml_inv * unit_cost * daily_holding_rate
            ml_holding_costs += daily_ml_hold

            if is_perishable and ml_inv > (actual_demand * 2):
                ml_spoilage_units += (ml_inv - actual_demand * 2) * spoilage_rate_perishables

    stockout_reduction_pct = (
        ((static_stockout_events - ml_stockout_events) / static_stockout_events * 100)
        if static_stockout_events > 0 else 0.0
    )
    holding_cost_reduction_pct = (
        ((static_holding_costs - ml_holding_costs) / static_holding_costs * 100)
        if static_holding_costs > 0 else 0.0
    )
    spoilage_reduction_pct = (
        ((static_spoilage_units - ml_spoilage_units) / static_spoilage_units * 100)
        if static_spoilage_units > 0 else 0.0
    )

    print("\n" + "=" * 50)
    print("BUSINESS KPI IMPACT SIMULATION RESULTS:")
    print("=" * 50)
    print(f"1. STOCKOUT PREVENTION:")
    print(f"   - Static Policy Stockout Events : {static_stockout_events}")
    print(f"   - ML Dynamic Policy Stockouts  : {ml_stockout_events}")
    print(f"   -> Stockouts Prevented         : {stockout_reduction_pct:.1f}% reduction")
    print()
    print(f"2. HOLDING CAPITAL / INVENTORY COSTS:")
    print(f"   - Static Policy Holding Cost   : Rs. {static_holding_costs:,.2f}")
    print(f"   - ML Dynamic Policy Holding    : Rs. {ml_holding_costs:,.2f}")
    print(f"   -> Holding Cost Savings        : {holding_cost_reduction_pct:.1f}% reduction")
    print()
    print(f"3. PERISHABLE SPOILAGE & SHRINKAGE:")
    print(f"   - Static Policy Spoilage Units : {static_spoilage_units:,.1f} units")
    print(f"   - ML Dynamic Policy Spoilage   : {ml_spoilage_units:,.1f} units")
    print(f"   -> Spoilage Reduction          : {spoilage_reduction_pct:.1f}% reduction")
    print("=" * 50)

    benchmarks = load_model_benchmarks(outputs_dir)
    comparison_records = []

    if "ARIMA" in benchmarks:
        comparison_records.append({
            "Model Architecture": "ARIMA Baseline (5,1,0)",
            "Role": "Benchmark",
            "RMSE": benchmarks["ARIMA"].get("avg_rmse"),
            "MAPE (%)": f"{benchmarks['ARIMA'].get('avg_mape')}%",
            "Status": "Completed",
        })

    if "XGBoost" in benchmarks:
        comparison_records.append({
            "Model Architecture": "XGBoost Regressor",
            "Role": "Core Engine",
            "RMSE": benchmarks["XGBoost"]["metrics"]["rmse"],
            "MAPE (%)": f"{benchmarks['XGBoost']['metrics']['mape']}%",
            "Status": "Completed",
        })

    if "LSTM" in benchmarks:
        comparison_records.append({
            "Model Architecture": "LSTM Neural Network",
            "Role": "Stretch Goal",
            "RMSE": benchmarks["LSTM"]["metrics"]["rmse"],
            "MAPE (%)": f"{benchmarks['LSTM']['metrics']['mape']}%",
            "Status": "Completed",
        })

    comparison_df = pd.DataFrame(comparison_records)
    print("\nTECHNICAL FORECAST ACCURACY BENCHMARKS:")
    print(comparison_df.to_string(index=False))

    kpi_report = {
        "technical_benchmarks": comparison_records,
        "business_kpis": {
            "stockout_reduction_pct": round(stockout_reduction_pct, 2),
            "holding_cost_reduction_pct": round(holding_cost_reduction_pct, 2),
            "spoilage_reduction_pct": round(spoilage_reduction_pct, 2),
            "static_stockouts": static_stockout_events,
            "ml_stockouts": ml_stockout_events,
            "static_holding_cost_inr": round(static_holding_costs, 2),
            "ml_holding_cost_inr": round(ml_holding_costs, 2),
            "static_spoilage_units": round(static_spoilage_units, 1),
            "ml_spoilage_units": round(ml_spoilage_units, 1),
        }
    }

    out_file = os.path.join(outputs_dir, "kpi_simulation_report.json")
    with open(out_file, "w") as f:
        json.dump(kpi_report, f, indent=2)

    print(f"\n[KPI Simulation] Full business KPI report saved -> {out_file}")
    print("=" * 60)
    return kpi_report


if __name__ == "__main__":
    run_inventory_business_simulation()
