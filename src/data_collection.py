import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_nifty50(start="2019-01-01", end="2024-12-31"):
    print("Downloading NIFTY-50 data...")
    df = yf.download("^NSEI", start=start, end=end)
    df = df[['Close']]
    df.columns = ['Price']
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df = df.dropna()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/nifty_prices.csv")
    print(f"✅ Saved {len(df)} rows to data/nifty_prices.csv")
    return df

if __name__ == "__main__":
    df = download_nifty50()
    print(df.tail())