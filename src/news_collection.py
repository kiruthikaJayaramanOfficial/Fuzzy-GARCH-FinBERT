import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def fetch_news(query="NIFTY OR NSE OR Indian stock market",
               days_back=30, language="en"):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY not found in .env file!")

    all_articles = []
    for i in range(days_back):
        date = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={query}&from={date}&to={date}"
            f"&language={language}&sortBy=relevancy"
            f"&apiKey={api_key}"
        )
        response = requests.get(url)
        data = response.json()

        if data.get("status") == "ok":
            for article in data.get("articles", []):
                all_articles.append({
                    "date": date,
                    "headline": article.get("title", ""),
                    "source": article.get("source", {}).get("name", "")
                })

    df = pd.DataFrame(all_articles)
    df = df[df["headline"].notna() & (df["headline"] != "")]
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/news_headlines.csv", index=False)
    print(f"✅ Saved {len(df)} headlines to data/news_headlines.csv")
    return df

if __name__ == "__main__":
    df = fetch_news(days_back=30)
    print(df.head())