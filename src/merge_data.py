import pandas as pd
import os

def merge_price_news():
    print("Loading price and news data...")
    
    prices = pd.read_csv("data/nifty_prices.csv", index_col="Date", parse_dates=True)
    news = pd.read_csv("data/news_headlines.csv", parse_dates=["date"])

    # Group headlines by date — join all headlines for same day
    news_grouped = news.groupby("date")["headline"].apply(
        lambda x: " | ".join(x)
    ).reset_index()
    news_grouped.columns = ["Date", "headlines"]
    news_grouped["Date"] = pd.to_datetime(news_grouped["Date"])
    news_grouped = news_grouped.set_index("Date")

    # Merge on date — forward fill missing news days
    merged = prices.join(news_grouped, how="left")
    merged["headlines"] = merged["headlines"].fillna("no news")

    os.makedirs("data", exist_ok=True)
    merged.to_csv("data/merged_data.csv")
    print(f"✅ Saved {len(merged)} rows to data/merged_data.csv")
    print(merged[["Price", "log_return", "headlines"]].tail())
    return merged

if __name__ == "__main__":
    merge_price_news()