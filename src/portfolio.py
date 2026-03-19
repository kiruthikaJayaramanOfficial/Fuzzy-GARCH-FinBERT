import pandas as pd
import numpy as np
import os

def simulate_portfolio():
    print("Loading fuzzy index...")
    df = pd.read_csv("data/fuzzy_index.csv",
                     index_col="Date", parse_dates=True)
    df = df.dropna(subset=["log_return", "mu", "nu", "pi"])

    # --- Strategy 1: Baseline (always 100% invested) ---
    df["exposure_baseline"] = 1.0

    # --- Strategy 2: Fuzzy exposure using WIFCM mu/nu/pi ---
    # mu > 0.6 → high confidence bullish → full exposure
    # pi > 0.4 → high uncertainty → reduce exposure
    # nu > 0.6 → high confidence bearish → exit market
    def fuzzy_exposure(row):
        if row["pi"] > 0.4:
            return 0.5  # uncertain → half exposure
        elif row["mu"] >= 0.6:
            return 1.0  # bullish → full exposure
        elif row["nu"] >= 0.6:
            return 0.0  # bearish → exit
        else:
            return 1 - row["pi"]  # graded exposure

    df["exposure_fuzzy"] = df.apply(fuzzy_exposure, axis=1)

    # --- Compute daily returns for each strategy ---
    df["return_baseline"] = df["exposure_baseline"] * df["log_return"]
    df["return_fuzzy"]    = df["exposure_fuzzy"]    * df["log_return"]

    # --- Cumulative returns ---
    df["cumulative_baseline"] = df["return_baseline"].cumsum()
    df["cumulative_fuzzy"]    = df["return_fuzzy"].cumsum()

    # --- Sharpe Ratio (annualized) ---
    def sharpe(returns, periods=252):
        mean   = returns.mean() * periods
        std    = returns.std()  * np.sqrt(periods)
        return mean / std if std > 0 else 0

    sharpe_baseline = sharpe(df["return_baseline"])
    sharpe_fuzzy    = sharpe(df["return_fuzzy"])

    print(f"\n--- Portfolio Results ---")
    print(f"  Baseline Sharpe : {sharpe_baseline:.4f}")
    print(f"  Fuzzy    Sharpe : {sharpe_fuzzy:.4f}")
    print(f"  Improvement     : {((sharpe_fuzzy/sharpe_baseline)-1)*100:.2f}%")

    # --- Save results ---
    os.makedirs("data", exist_ok=True)
    df[[
        "log_return", "mu", "nu", "pi",
        "exposure_baseline", "exposure_fuzzy",
        "return_baseline", "return_fuzzy",
        "cumulative_baseline", "cumulative_fuzzy"
    ]].to_csv("data/portfolio_results.csv")

    summary = pd.DataFrame({
        "Strategy": ["Baseline", "Fuzzy-WIFCM"],
        "Sharpe":   [sharpe_baseline, sharpe_fuzzy],
        "Final_Return": [
            df["cumulative_baseline"].iloc[-1],
            df["cumulative_fuzzy"].iloc[-1]
        ]
    })
    summary.to_csv("data/portfolio_summary.csv", index=False)
    print("\n--- Summary ---")
    print(summary.to_string(index=False))
    print("✅ Saved portfolio results")
    return df, summary

if __name__ == "__main__":
    simulate_portfolio()