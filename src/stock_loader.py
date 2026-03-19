import pandas as pd
import numpy as np
import os
import glob

STOCK_FILES = {
    "ADANI_PORTS":    "ADANI_PORTS.csv",
    "BAJAJ_AUTO":     "BAJAJ AUTO.csv",
    "BAJAJ_FINANCE":  "BAJAJ_FINANCE.csv",
    "BAJAJ_FINSERV":  "BAJAJ_FINSERV.csv",
    "BRITANNIA":      "BRITANNIA.csv",
    "DIVIS_LAB":      "DIVIS LAB.csv",
    "HDFC_BANK":      "HDFC_BANK.csv",
    "HDFC_LIFE":      "HDFC_LIFE.csv",
    "HINDALCO":       "HINDALCO.csv",
    "INDUS_IND":      "INDUS INDUSTRIES.csv",
    "INFOSYS":        "INFOSYS.csv",
    "ITC":            "ITC.csv",
    "KOTAK":          "KOTAK_MAHINDRA.csv",
    "RELIANCE":       "RELIANCE.csv",
    "SBI":            "SBI_BANK.csv",
    "SBI_LIFE":       "SBI_LIFE.csv",
    "TATA_CONSUMER":  "TATA CONSUMER PRODUCTS.csv",
    "TITAN":          "TITAN.csv",
    "UPL":            "UPL.csv",
    "WIPRO":          "WIPRO.csv",
}

SECTORS = {
    "HDFC_BANK": "Banking",   "KOTAK": "Banking",
    "SBI": "Banking",         "INDUS_IND": "Banking",
    "HDFC_LIFE": "Insurance", "SBI_LIFE": "Insurance",
    "BAJAJ_FINANCE": "NBFC",  "BAJAJ_FINSERV": "NBFC",
    "INFOSYS": "IT",          "WIPRO": "IT",
    "RELIANCE": "Energy",     "HINDALCO": "Metals",
    "ADANI_PORTS": "Infra",   "BAJAJ_AUTO": "Auto",
    "TATA_CONSUMER": "FMCG",  "ITC": "FMCG",
    "BRITANNIA": "FMCG",      "TITAN": "Consumer",
    "DIVIS_LAB": "Pharma",    "UPL": "Agro",
}

def load_all_stocks(data_dir="data/stocks"):
    """Load all stock CSVs → returns dict of DataFrames."""
    stocks = {}
    for name, filename in STOCK_FILES.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().title() for c in df.columns]
            date_col  = [c for c in df.columns
                         if 'date' in c.lower()][0]
            close_col = [c for c in df.columns
                         if 'close' in c.lower()
                         and 'adj' not in c.lower()][0]
            df[date_col] = pd.to_datetime(
                df[date_col], dayfirst=True,
                format='mixed'
            )
            df = df.rename(columns={
                date_col: 'Date',
                close_col: 'Close'
            })
            df = df[['Date','Close']].dropna()
            df = df.sort_values('Date').set_index('Date')
            df['log_return'] = np.log(
                df['Close'] / df['Close'].shift(1)
            )
            stocks[name] = df
        except Exception as e:
            print(f"  ⚠️ {name}: {e}")
    print(f"✅ Loaded {len(stocks)} stocks")
    return stocks

def get_stock_metrics(stocks, start="2022-01-01",
                      end="2023-06-23"):
    """Compute return/vol metrics for event window."""
    metrics = {}
    for name, df in stocks.items():
        window = df.loc[start:end, 'log_return'].dropna()
        if len(window) < 10:
            continue
        metrics[name] = {
            "total_return": float(window.sum()),
            "volatility":   float(window.std() * np.sqrt(252)),
            "sharpe":       float(
                window.mean()*252 /
                (window.std()*np.sqrt(252)+1e-9)
            ),
            "sector": SECTORS.get(name, "Other")
        }
    return metrics

if __name__ == "__main__":
    stocks = load_all_stocks()
    metrics = get_stock_metrics(stocks)
    for k,v in list(metrics.items())[:5]:
        print(f"{k}: return={v['total_return']:.3f} "
              f"vol={v['volatility']:.3f} "
              f"sharpe={v['sharpe']:.3f}")