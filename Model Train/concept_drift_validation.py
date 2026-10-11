"""
concept_drift_validation.py
Mitigation for Concept Drift: Walk-Forward Rolling Validation Windows.

As specified in Phase 1 (PDF Slide 8 & README):
"Rolling validation windows to keep models current with shifting patterns."

In retail demand forecasting, consumer buying patterns continuously drift due to
seasonal climate changes, festival periods, and evolving purchase cycles.
This module validates model adaptation across consecutive temporal walk-forward windows.
Outputs are saved to 'outputs/'.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from xgboost_model import prepare_features, calculate_metrics


def run_rolling_window_validation(datasets_dir="datasets", outputs_dir="outputs"):
    print("=" * 60)
    print("RUNNING CONCEPT DRIFT VALIDATION: ROLLING WALK-FORWARD WINDOWS")
    print("=" * 60)

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    csv_path = os.path.join(datasets_dir, "xgb_training_ready.csv")
    if not os.path.exists(csv_path):
        csv_path = "xgb_training_ready.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}. Run preprocess_india_retail_store_data.py first.")

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    windows = [
        {"name": "Window 1 (Summer - May)", "train_end": "2024-04-30", "test_start": "2024-05-01", "test_end": "2024-05-31"},
        {"name": "Window 2 (Monsoon - Jul)", "train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-07-31"},
        {"name": "Window 3 (Festival Onset - Sep)", "train_end": "2024-08-31", "test_start": "2024-09-01", "test_end": "2024-09-30"},
        {"name": "Window 4 (Post-Diwali - Nov)", "train_end": "2024-10-31", "test_start": "2024-11-01", "test_end": "2024-11-30"},
        {"name": "Window 5 (Winter Holidays - Dec)", "train_end": "2024-11-30", "test_start": "2024-12-01", "test_end": "2024-12-30"},
    ]

    has_xgboost = False
    try:
        import xgboost as xgb
        has_xgboost = True
    except ImportError:
        pass

    results = []

    for w in windows:
        train_df = df[df["Date"] <= pd.to_datetime(w["train_end"])].copy()
        test_df = df[(df["Date"] >= pd.to_datetime(w["test_start"])) & (df["Date"] <= pd.to_datetime(w["test_end"]))].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)

        if has_xgboost:
            model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.06,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)
            y_pred = np.maximum(model.predict(X_test), 0.0)
        else:
            X_tr = X_train.values.astype(float)
            y_tr = y_train.astype(float)
            X_te = X_test.values.astype(float)

            mean_x = np.mean(X_tr, axis=0)
            std_x = np.std(X_tr, axis=0) + 1e-7
            X_tr_norm = (X_tr - mean_x) / std_x
            X_te_norm = (X_te - mean_x) / std_x

            lmbda = 10.0
            XTX = np.dot(X_tr_norm.T, X_tr_norm) + lmbda * np.eye(X_tr_norm.shape[1])
            XTy = np.dot(X_tr_norm.T, y_tr)
            weights = np.linalg.solve(XTX, XTy)
            y_pred = np.maximum(np.dot(X_te_norm, weights), 0.0)

        metrics = calculate_metrics(y_test, y_pred)
        results.append({
            "Window": w["name"],
            "Train End": w["train_end"],
            "Test Period": f"{w['test_start']} to {w['test_end']}",
            "Test Records": len(test_df),
            "RMSE": metrics["rmse"],
            "MAPE (%)": f"{metrics['mape']}%",
            "MAE": metrics["mae"],
            "R2": metrics["r2"],
        })

    results_df = pd.DataFrame(results)
    print("\nWalk-Forward Rolling Window Validation Results:")
    print(results_df.to_string(index=False))

    out_csv = os.path.join(outputs_dir, "concept_drift_report.csv")
    results_df.to_csv(out_csv, index=False)
    # Also mirror to datasets/concept_drift_report.csv
    results_df.to_csv(os.path.join(datasets_dir, "concept_drift_report.csv"), index=False)

    print(f"\n[Concept Drift] Walk-forward validation output saved -> {out_csv}")
    print("=" * 60)
    return results_df


if __name__ == "__main__":
    run_rolling_window_validation()
