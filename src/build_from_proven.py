import pandas as pd
import numpy as np
import os
import mlflow
import warnings
warnings.filterwarnings("ignore")

def build_combined_pipeline():
    print("=== Loading proven Colab results ===")
    df = pd.read_csv("data/tableau_master_data_FINAL.csv",
                     parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    print(f"✅ Loaded {len(df)} rows: {df.index[0].date()} → {df.index[-1].date()}")

    # ── Step 1: Apply WIFCM mu/nu/pi on top of proven sentiment ──
    print("\n=== Applying WIFCM mu/nu/pi ===")
    alpha, beta = 1.5, 0.5

    sent = df["Sentiment_Rolled"].fillna(0).values
    sent_clipped = np.clip(sent, -1, 1)

    pos = np.clip( sent_clipped, 0, 1)
    neg = np.clip(-sent_clipped, 0, 1)
    neu = np.clip(1 - pos - neg, 0.05, 1)

    total = pos + neg + neu + 1e-9
    pos /= total
    neg /= total

    def membership(r):
        num = 1 - np.power(r + 1e-9, alpha)
        den = 1 - np.power(beta * r + 1e-9, alpha)
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)
        return np.clip(np.power(np.clip(num/den, 0, 1), 1.0/alpha), 0, 1)

    mu = membership(pos)
    nu = membership(neg)
    total_mn  = mu + nu
    violation = total_mn > 1
    mu[violation] = mu[violation] / total_mn[violation]
    nu[violation] = nu[violation] / total_mn[violation]
    pi = np.clip(1 - mu - nu, 0, 1)

    df["mu"]              = mu
    df["nu"]              = nu
    df["pi"]              = pi
    df["fuzzy_sentiment"] = (mu - nu) * (1 - pi)
    df["fuzzy_index_5d"]  = df["fuzzy_sentiment"].rolling(5).mean()

    print(f"  Bullish days  (mu>0.55): {(mu>0.55).sum()}")
    print(f"  Bearish days  (nu>0.55): {(nu>0.55).sum()}")
    print(f"  Uncertain days (pi>0.3): {(pi>0.3).sum()}")

    # ── Step 2: GARCH-X vs Baseline MAE (proven numbers) ──
    print("\n=== Computing proven MAE metrics ===")
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    valid = df[["Realized_Vol","GARCH_Forecast_Vol",
                "GARCHX_Forecast_Vol"]].dropna()

    mae_baseline = mean_absolute_error(valid["Realized_Vol"],
                                       valid["GARCH_Forecast_Vol"])
    mae_garchx   = mean_absolute_error(valid["Realized_Vol"],
                                       valid["GARCHX_Forecast_Vol"])
    rmse_baseline = np.sqrt(mean_squared_error(valid["Realized_Vol"],
                                               valid["GARCH_Forecast_Vol"]))
    rmse_garchx   = np.sqrt(mean_squared_error(valid["Realized_Vol"],
                                               valid["GARCHX_Forecast_Vol"]))
    mae_improvement  = (mae_baseline  - mae_garchx)  / mae_baseline  * 100
    rmse_improvement = (rmse_baseline - rmse_garchx) / rmse_baseline * 100

    print(f"  GARCH    MAE : {mae_baseline:.4f}")
    print(f"  GARCH-X  MAE : {mae_garchx:.4f}  ({mae_improvement:+.2f}%)")
    print(f"  GARCH    RMSE: {rmse_baseline:.4f}")
    print(f"  GARCH-X  RMSE: {rmse_garchx:.4f}  ({rmse_improvement:+.2f}%)")

    # ── Step 3: Fuzzy portfolio using WIFCM mu/nu/pi ──
    print("\n=== Fuzzy portfolio simulation ===")
    df["return"] = df["Log_Return_pct"]

    def fuzzy_exposure(row):
        if row["pi"] > 0.4:   return 0.5
        elif row["mu"] >= 0.6: return 1.0
        elif row["nu"] >= 0.6: return 0.0
        else:                  return 1 - row["pi"]

    df["exposure_fuzzy"]    = df.apply(fuzzy_exposure, axis=1)
    df["exposure_baseline"] = 1.0
    df["return_fuzzy"]      = df["exposure_fuzzy"]    * df["return"]
    df["return_baseline"]   = df["exposure_baseline"] * df["return"]
    df["cumulative_fuzzy"]  = df["return_fuzzy"].cumsum()
    df["cumulative_baseline"] = df["return_baseline"].cumsum()

    def sharpe(returns, periods=252):
        r = returns.dropna()
        mean = r.mean() * periods
        std  = r.std()  * np.sqrt(periods)
        return mean / std if std > 0 else 0

    sharpe_baseline = sharpe(df["return_baseline"])
    sharpe_fuzzy    = sharpe(df["return_fuzzy"])
    improvement     = ((sharpe_fuzzy/sharpe_baseline)-1)*100 \
                      if sharpe_baseline != 0 else 0

    print(f"  Baseline Sharpe : {sharpe_baseline:.4f}")
    print(f"  Fuzzy    Sharpe : {sharpe_fuzzy:.4f}")
    print(f"  Improvement     : {improvement:+.2f}%")

    # ── Step 4: Log to MLflow ──
    print("\n=== Logging to MLflow ===")
    mlflow.set_experiment("Fuzzy-GARCH-Combined")
    with mlflow.start_run(run_name="combined_wifcm_garchx"):
        mlflow.log_param("alpha",        alpha)
        mlflow.log_param("beta",         beta)
        mlflow.log_param("news_source",  "GDELT")
        mlflow.log_param("data_rows",    len(df))
        mlflow.log_metric("MAE_baseline",     mae_baseline)
        mlflow.log_metric("MAE_garchx",       mae_garchx)
        mlflow.log_metric("MAE_improvement",  mae_improvement)
        mlflow.log_metric("RMSE_improvement", rmse_improvement)
        mlflow.log_metric("Sharpe_baseline",  sharpe_baseline)
        mlflow.log_metric("Sharpe_fuzzy",     sharpe_fuzzy)
        mlflow.log_metric("Sharpe_improvement", improvement)
    print("✅ Logged to MLflow")

    # ── Step 5: Save all results ──
    os.makedirs("data/forecasts", exist_ok=True)
    df.to_csv("data/fuzzy_index.csv")

    pd.DataFrame({
        "Model": ["GARCH Baseline","GARCH-X+WIFCM"],
        "MAE":   [mae_baseline, mae_garchx],
        "RMSE":  [rmse_baseline, rmse_garchx],
        "Improvement": [0, mae_improvement]
    }).to_csv("data/forecasts/model_comparison.csv", index=False)

    pd.DataFrame({
        "Strategy": ["Baseline","Fuzzy-WIFCM"],
        "Sharpe":   [sharpe_baseline, sharpe_fuzzy],
        "Final_Return": [
            df["cumulative_baseline"].iloc[-1],
            df["cumulative_fuzzy"].iloc[-1]
        ]
    }).to_csv("data/portfolio_summary.csv", index=False)

    df[[
        "return","mu","nu","pi",
        "exposure_baseline","exposure_fuzzy",
        "return_baseline","return_fuzzy",
        "cumulative_baseline","cumulative_fuzzy"
    ]].to_csv("data/portfolio_results.csv")

    print("\n✅ All results saved!")
    print(f"""
╔══════════════════════════════════════════════╗
║     FINAL RESULTS SUMMARY                   ║
╠══════════════════════════════════════════════╣
  VOLATILITY FORECASTING
  GARCH    MAE  : {mae_baseline:.4f}
  GARCH-X  MAE  : {mae_garchx:.4f}  ({mae_improvement:+.2f}%)
  GARCH    RMSE : {rmse_baseline:.4f}
  GARCH-X  RMSE : {rmse_garchx:.4f}  ({rmse_improvement:+.2f}%)

  PORTFOLIO (WIFCM mu/nu/pi)
  Baseline Sharpe : {sharpe_baseline:.4f}
  Fuzzy    Sharpe : {sharpe_fuzzy:.4f}  ({improvement:+.2f}%)

  NEWS SOURCE : GDELT (real historical)
  WIFCM alpha : {alpha}, beta : {beta}
╚══════════════════════════════════════════════╝
""")
    return df

if __name__ == "__main__":
    build_combined_pipeline()