"""
run_phase1_pipeline.py
End-to-End Orchestrator for Phase 1: Data Engineering & Model Training.

Executes all Phase 1 components in sequence:
1. Data Preprocessing & Contextual Enrichment (Holidays, Weather, Censored Demand, Lags)
   -> Generates feature-engineered datasets into 'datasets/'
2. ARIMA Time-Series Baseline Benchmark
   -> Generates benchmark metrics into 'outputs/'
3. XGBoost Core Demand Forecasting Engine
   -> Generates model, evaluation results into 'outputs/' and prediction dataset into 'datasets/'
4. Concept Drift Rolling Validation Windows
   -> Generates drift report into 'outputs/' and 'datasets/'
5. LSTM Neural Network Stretch Goal
   -> Generates weights and benchmark into 'outputs/'
6. Business KPI Impact Simulation (Holding Costs, Spoilage, Stockouts)
   -> Generates comparative KPI report into 'outputs/'
"""

import os
import sys
import time
from preprocess_india_retail_store_data import preprocess_all
from ARIMA_model import run_arima_benchmark
from xgboost_model import train_and_evaluate_xgboost
from concept_drift_validation import run_rolling_window_validation
from lstm_model import run_lstm_model
from evaluate_and_simulate_kpis import run_inventory_business_simulation


def run_full_pipeline(datasets_dir="datasets", outputs_dir="outputs"):
    start_time = time.time()
    print("=" * 70)
    print("  INTELLIGENT DEMAND FORECASTING & INVENTORY OPTIMIZATION")
    print("           PHASE 1: FULL PIPELINE EXECUTION")
    print("=" * 70)

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"Target Datasets Directory : {os.path.abspath(datasets_dir)}")
    print(f"Target Outputs Directory  : {os.path.abspath(outputs_dir)}\n")

    # Step 1: Preprocessing & Contextual Enrichment (saves to datasets/)
    print("\n[STEP 1/6] Ingestion, Weather & Holiday Enrichment, Censored Demand Recovery...")
    preprocess_all(datasets_dir=datasets_dir)

    # Step 2: ARIMA Baseline (saves to outputs/)
    print("\n[STEP 2/6] Statistical ARIMA Baseline Modeling...")
    run_arima_benchmark(datasets_dir=datasets_dir, outputs_dir=outputs_dir)

    # Step 3: Core XGBoost Engine (saves to outputs/ and datasets/)
    print("\n[STEP 3/6] Multivariate XGBoost Engine Training & Evaluation...")
    train_and_evaluate_xgboost(datasets_dir=datasets_dir, outputs_dir=outputs_dir)

    # Step 4: Concept Drift Mitigation (saves to outputs/ and datasets/)
    print("\n[STEP 4/6] Walk-Forward Rolling Window Validation (Concept Drift)...")
    run_rolling_window_validation(datasets_dir=datasets_dir, outputs_dir=outputs_dir)

    # Step 5: LSTM Stretch Goal (saves to outputs/)
    print("\n[STEP 5/6] Sequential LSTM Neural Network Modeling (Stretch Goal)...")
    run_lstm_model(datasets_dir=datasets_dir, outputs_dir=outputs_dir)

    # Step 6: KPI Simulation (saves to outputs/)
    print("\n[STEP 6/6] Business KPI Simulation & Model Comparison Report...")
    run_inventory_business_simulation(datasets_dir=datasets_dir, outputs_dir=outputs_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PHASE 1 PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print(f"  - Datasets saved in: {os.path.abspath(datasets_dir)}")
    print(f"  - Outputs saved in:  {os.path.abspath(outputs_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
