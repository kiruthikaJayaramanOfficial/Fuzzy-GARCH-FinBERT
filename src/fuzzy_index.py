import pandas as pd
import numpy as np
import os

def compute_wifcm_fuzzy_index(alpha=1.5, beta=0.5):
    """
    Apply WIFCM membership function from research paper to finance.
    pos_score  -> mu    (membership/bullish)
    neg_score  -> nu    (non-membership/bearish)
    hesitancy  -> pi    (uncertainty/hold cash)
    """
    print("Loading sentiment scores...")
    df = pd.read_csv("data/sentiment_scores.csv",
                     index_col="Date", parse_dates=True)

    pos = df["pos_score"].values
    neg = df["neg_score"].values
    neu = df["neu_score"].values

    # Step 1: Compute relative distance r (from your paper)
    total = pos + neg + neu + 1e-9
    r_pos = pos / total
    r_neg = neg / total

    # Step 2: Apply your novel membership function
    # M = ((1 - r^alpha) / (1 - (beta*r)^alpha))^(1/alpha)
    def membership(r, alpha, beta):
        numerator   = 1 - np.power(r + 1e-9, alpha)
        denominator = 1 - np.power(beta * r + 1e-9, alpha)
        denominator = np.where(np.abs(denominator) < 1e-9, 1e-9, denominator)
        M = np.power(np.clip(numerator / denominator, 0, 1), 1.0 / alpha)
        return np.clip(M, 0, 1)

    # Step 3: Compute mu (bullish), nu (bearish)
    mu = membership(r_pos, alpha, beta)   # bullish membership
    nu = membership(r_neg, alpha, beta)   # bearish non-membership

    # Step 4: Normalize so mu + nu <= 1
    total_mn = mu + nu
    violation = total_mn > 1
    mu[violation] = mu[violation] / total_mn[violation]
    nu[violation] = nu[violation] / total_mn[violation]

    # Step 5: Hesitancy pi = 1 - mu - nu (uncertainty)
    pi = 1 - mu - nu
    pi = np.clip(pi, 0, 1)

    # Step 6: Daily fuzzy sentiment index
    # bullish=1, bearish=-1, weighted by hesitancy
    daily_mu  = mu           # bullish degree
    daily_nu  = nu           # bearish degree
    daily_pi  = pi           # uncertainty degree

    df["mu"]       = daily_mu
    df["nu"]       = daily_nu
    df["pi"]       = daily_pi

    # Fuzzy sentiment: +mu (bull) - nu (bear), scaled by (1-pi)
    df["fuzzy_sentiment"] = (mu - nu) * (1 - pi)

    # Rolling 5-day fuzzy index
    df["fuzzy_index_5d"] = df["fuzzy_sentiment"].rolling(5).mean()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/fuzzy_index.csv")
    print("✅ Saved fuzzy index to data/fuzzy_index.csv")
    print(df[["mu", "nu", "pi", "fuzzy_sentiment", "fuzzy_index_5d"]].tail(10))
    return df

if __name__ == "__main__":
    df = compute_wifcm_fuzzy_index()