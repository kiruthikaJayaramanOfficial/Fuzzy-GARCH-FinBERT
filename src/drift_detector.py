import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
import os
import json

def detect_drift(threshold=0.15):
    print("Loading fuzzy index for drift detection...")
    df = pd.read_csv("data/fuzzy_index.csv",
                     index_col="Date", parse_dates=True)
    df = df[["mu", "nu", "pi", "fuzzy_sentiment"]].dropna()

    # Split: reference = first 80%, current = last 20%
    split     = int(len(df) * 0.8)
    reference = df.iloc[:split].reset_index(drop=True)
    current   = df.iloc[split:].reset_index(drop=True)

    print(f"  Reference: {len(reference)} rows")
    print(f"  Current  : {len(current)} rows")

    # Build Evidently datasets
    definition  = DataDefinition()
    ref_dataset = Dataset.from_pandas(reference, data_definition=definition)
    cur_dataset = Dataset.from_pandas(current,   data_definition=definition)

    # Run drift report
    report  = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_dataset,
                         current_data=cur_dataset)

    # Save HTML
    os.makedirs("eval", exist_ok=True)
    my_eval.save_html("eval/drift_report.html")
    print("✅ Drift report saved to eval/drift_report.html")

    # Extract drift score safely
    try:
        result_dict   = my_eval.dict()
        metrics       = result_dict.get("metrics", [])
        share_drifted = 0.0
        for m in metrics:
            val = m.get("value", {})
            if isinstance(val, dict) and "share_of_drifted_columns" in val:
                share_drifted = float(val["share_of_drifted_columns"])
                break
    except Exception:
        share_drifted = 0.0

    drift_detected = share_drifted > threshold

    print(f"\n--- Drift Results ---")
    print(f"  Share drifted cols : {share_drifted:.2%}")
    print(f"  Drift detected     : {drift_detected}")

    summary = {
        "drift_detected" : drift_detected,
        "share_drifted"  : share_drifted,
        "threshold"      : threshold,
        "trigger_retrain": drift_detected
    }
    with open("eval/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Trigger retrain    : {summary['trigger_retrain']}")
    print("✅ Saved to eval/drift_summary.json")
    return summary

if __name__ == "__main__":
    detect_drift()