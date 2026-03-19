import pandas as pd
import numpy as np
from transformers import pipeline
import os

def get_finbert_scores(batch_size=32):
    print("Loading FinBERT model...")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
        device=-1  # CPU
    )

    df = pd.read_csv("data/merged_data.csv",
                     index_col="Date", parse_dates=True)
    headlines = df["headlines"].tolist()

    pos_scores, neg_scores, neu_scores = [], [], []

    print(f"Scoring {len(headlines)} entries...")
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        # Truncate long texts
        batch = [str(h)[:512] for h in batch]
        results = finbert(batch)
        for result in results:
            scores = {r["label"]: r["score"] for r in result}
            pos_scores.append(scores.get("positive", 0))
            neg_scores.append(scores.get("negative", 0))
            neu_scores.append(scores.get("neutral", 0))

        if i % 200 == 0:
            print(f"  Processed {i}/{len(headlines)}...")

    df["pos_score"] = pos_scores
    df["neg_score"] = neg_scores
    df["neu_score"] = neu_scores
    df.to_csv("data/sentiment_scores.csv")
    print(f"✅ Saved sentiment scores to data/sentiment_scores.csv")
    return df

if __name__ == "__main__":
    df = get_finbert_scores()
    print(df[["pos_score", "neg_score", "neu_score"]].tail())