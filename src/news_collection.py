import requests
import pandas as pd
import time
import os

QUERIES = [
    'India stock market Sensex Nifty',
    'BSE NSE India equity market',
    'India economic growth GDP inflation',
    'RBI interest rate India rupee',
    'Reliance Infosys TCS Wipro HDFC India',
    'India budget finance ministry rupee',
]

def fetch_gdelt(query, start_date, end_date, max_records=250):
    """Fetch from GDELT 2.0 — free, no API key, full history."""
    params = {
        'query'        : query,
        'mode'         : 'artlist',
        'format'       : 'json',
        'maxrecords'   : max_records,
        'startdatetime': start_date.replace('-','') + '000000',
        'enddatetime'  : end_date.replace('-','')   + '235959',
        'sort'         : 'DateDesc',
    }
    try:
        r = requests.get(
            'https://api.gdeltproject.org/api/v2/doc/doc',
            params=params, timeout=20
        )
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for a in r.json().get('articles', []):
            pub   = a.get('seendate', '')
            title = (a.get('title') or '').strip()
            if pub and title:
                try:
                    dt = pd.to_datetime(
                        pub, format='%Y%m%dT%H%M%SZ'
                    ).normalize()
                except:
                    dt = pd.NaT
                rows.append({'date': dt, 'headline': title,
                             'source': a.get('domain','')})
        df = pd.DataFrame(rows).dropna(subset=['date','headline'])
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f'  GDELT error: {e}')
        return pd.DataFrame()

def fetch_news(start_date="2019-01-01", end_date="2024-12-31"):
    print(f"Fetching GDELT news: {start_date} → {end_date}")
    monthly_starts = pd.date_range(start_date, end_date, freq='MS')
    monthly_ends   = [
        min(s + pd.offsets.MonthEnd(1), pd.to_datetime(end_date))
        for s in monthly_starts
    ]

    raw_frames = []
    total = len(QUERIES) * len(monthly_starts)
    done  = 0

    for q in QUERIES:
        print(f'\n  Query: "{q}"')
        for s, e in zip(monthly_starts, monthly_ends):
            chunk = fetch_gdelt(
                q,
                s.strftime('%Y-%m-%d'),
                e.strftime('%Y-%m-%d'),
                250
            )
            if not chunk.empty:
                raw_frames.append(chunk)
            done += 1
            if done % 10 == 0:
                print(f'    {done}/{total} batches done...')
            time.sleep(0.5)   # be polite to GDELT

    if not raw_frames:
        print("⚠️  No articles fetched!")
        return pd.DataFrame()

    df = pd.concat(raw_frames, ignore_index=True)
    df = (df
          .drop_duplicates(subset='headline')
          .sort_values('date')
          .reset_index(drop=True))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/news_headlines.csv", index=False)
    print(f"\n✅ Saved {len(df)} headlines to data/news_headlines.csv")
    print(f"   Date range: {df['date'].min().date()} → "
          f"{df['date'].max().date()}")
    return df

if __name__ == "__main__":
    df = fetch_news()
    print(df.head())