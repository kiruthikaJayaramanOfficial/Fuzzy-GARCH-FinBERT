import pandas as pd
import numpy as np
from arch import arch_model
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings("ignore")

def fit_garch_models():
    print("Loading fuzzy index data...")
    df = pd.read_csv("data/fuzzy_index.csv",
                     index_col="Date", parse_dates=True)
    returns = df["log_return"].dropna() * 100  # scale for GARCH

    os.makedirs("data/forecasts", exist_ok=True)
    mlflow.set_experiment("Fuzzy-GARCH")

    results = {}

    # --- Model 1: Baseline GARCH(1,1) no sentiment ---
    with mlflow.start_run(run_name="baseline_garch"):
        print("\nFitting Baseline GARCH(1,1)...")
        m1 = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
        r1 = m1.fit(disp="off")
        forecasts1 = r1.forecast(horizon=1)
        vol1 = np.sqrt(forecasts1.variance.dropna().values.flatten())
        mae1 = float(np.mean(np.abs(
            r1.conditional_volatility - returns.std()
        )))
        mlflow.log_param("model", "baseline_garch")
        mlflow.log_metric("MAE", mae1)
        print(f"  Baseline MAE: {mae1:.4f}")
        results["baseline"] = {"mae": mae1, "vol": vol1}

    # --- Model 2: GARCH + hard sentiment threshold ---
    with mlflow.start_run(run_name="hard_threshold_garch"):
        print("Fitting Hard-Threshold GARCH...")
        df_aligned = df.loc[returns.index]
        sentiment = df_aligned["fuzzy_sentiment"].fillna(0)
        # Hard threshold: if sentiment > 0 → bullish=1, else bearish=-1
        hard_signal = np.where(sentiment > 0, 1, -1)
        returns_adj = returns + 0.01 * hard_signal
        m2 = arch_model(returns_adj, vol="Garch", p=1, q=1, dist="normal")
        r2 = m2.fit(disp="off")
        mae2 = float(np.mean(np.abs(
            r2.conditional_volatility - returns_adj.std()
        )))
        mlflow.log_param("model", "hard_threshold_garch")
        mlflow.log_metric("MAE", mae2)
        print(f"  Hard-threshold MAE: {mae2:.4f}")
        results["hard_threshold"] = {"mae": mae2}

    # --- Model 3: Fuzzy-GARCH using WIFCM mu/nu/pi ---
    with mlflow.start_run(run_name="fuzzy_garch"):
        print("Fitting Fuzzy-GARCH (WIFCM)...")
        mu  = df_aligned["mu"].fillna(0.5)
        nu  = df_aligned["nu"].fillna(0.5)
        pi  = df_aligned["pi"].fillna(0.0)
        # Graded adjustment: bullish reduces vol, bearish increases it
        # high pi (uncertainty) = hold cash = dampen signal
        fuzzy_adj = (mu - nu) * (1 - pi)
        returns_fuzzy = returns + 0.01 * fuzzy_adj
        m3 = arch_model(returns_fuzzy, vol="Garch", p=1, q=1, dist="normal")
        r3 = m3.fit(disp="off")
        mae3 = float(np.mean(np.abs(
            r3.conditional_volatility - returns_fuzzy.std()
        )))
        mlflow.log_param("model", "fuzzy_garch")
        mlflow.log_param("alpha", 1.5)
        mlflow.log_param("beta", 0.5)
        mlflow.log_metric("MAE", mae3)
        mlflow.log_metric("MAE_improvement_vs_baseline",
                          mae1 - mae3)
        print(f"  Fuzzy-GARCH MAE: {mae3:.4f}")
        results["fuzzy_garch"] = {"mae": mae3}

    # Save comparison
    comparison = pd.DataFrame({
        "Model":  ["Baseline", "Hard-Threshold", "Fuzzy-GARCH"],
        "MAE":    [results["baseline"]["mae"],
                   results["hard_threshold"]["mae"],
                   results["fuzzy_garch"]["mae"]]
    })
    comparison.to_csv("data/forecasts/model_comparison.csv", index=False)
    print("\n--- Model Comparison ---")
    print(comparison.to_string(index=False))
    print("✅ Saved to data/forecasts/model_comparison.csv")
    return results

if __name__ == "__main__":
    fit_garch_models()