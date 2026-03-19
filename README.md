# 📈 Fuzzy-GARCH-FinBERT Volatility Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red?style=flat-square&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-3.1-blue?style=flat-square)
![Evidently](https://img.shields.io/badge/Evidently_AI-0.7-purple?style=flat-square)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-green?style=flat-square&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**NIFTY-50 sentiment-volatility forecasting** · WIFCM μ/ν/π replaces hard thresholds · Full MLOps pipeline

[🚀 Live Dashboard](https://your-app.streamlit.app) · [📄 Research Paper](#research-foundation) · [📊 Results](#results)

</div>

---

## 🎯 What This Project Does

> **One sentence:** Reads real financial news daily, converts it into fuzzy confidence scores (bullish/bearish/uncertain), feeds them into a GARCH volatility model, and automatically adjusts portfolio exposure — beating the market benchmark by **39.78% Sharpe ratio**.

Traditional quant systems treat sentiment as binary: *bullish = buy, bearish = sell*. This project asks: **what about uncertainty?** What if the model doesn't know? Using Weighted Intuitionistic Fuzzy C-Means (WIFCM), every market day gets three graded degrees — and the portfolio acts proportionally.

---

## 📸 Dashboard Preview

| Page | Description |
|------|-------------|
| 📰 **Sentiment Intelligence** | Live μ/ν/π fuzzy degrees, 7-day rolled sentiment, regime distribution |
| 📉 **Volatility Forecast** | GARCH vs GARCH-X comparison, realized vol chart, MAE gauge |
| 💼 **Portfolio Simulator** | Interactive threshold sliders, cumulative returns, exposure chart |
| 🔬 **Model Explainer** | Component breakdown, membership function visualizer, conclusion |
| 🧪 **Portfolio Optimizer** | Pick market event → WIFCM scores 20 stocks → allocation with real returns |

---

## 📊 Results

### Volatility Forecasting

| Model | MAE | RMSE | Improvement |
|-------|-----|------|-------------|
| GARCH(1,1) Baseline | 1.1244 | 1.2131 | — |
| **GARCH-X + WIFCM** | **0.9965** | **1.0720** | **+11.37% MAE · +11.63% RMSE** |

### Portfolio Performance (2022–2023, 366 trading days)

| Strategy | Sharpe Ratio | Total Return | vs Baseline |
|----------|-------------|--------------|-------------|
| Buy & Hold | 0.7536 | 21.31% | — |
| **Fuzzy-WIFCM** | **1.0534** | **29.03%** | **+39.78% Sharpe** |

> **News coverage:** 100% of trading days via GDELT · No API key required · Full history back to 2000

---

## 🆚 Traditional vs WIFCM Approach

```
TRADITIONAL HARD THRESHOLD          WIFCM FUZZY APPROACH
─────────────────────────           ─────────────────────────────
Sentiment > 0  → BUY (100%)        μ = 0.72, ν = 0.18, π = 0.10
Sentiment < 0  → SELL (0%)         Signal = (0.72-0.18)×(1-0.10)
No middle ground                    Exposure = 48.6% (graded)
Ignores uncertainty                 π captures ambiguity
Overreacts to noise                 Dampened by hesitancy degree

Russia-Ukraine Feb 2022:
→ Hard: "bearish → full exit"      → Fuzzy: π=0.65 (uncertain)
→ Missed partial recovery           → 50% exposure, captured upside
```

**The key difference:** When π (hesitancy) is high, the portfolio automatically holds cash — not because of a rule, but because the math says *"we don't know enough to act fully."*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
│  GDELT API (free, historical) ──→ 5,731 real headlines         │
│  NIFTY-50 Stocks (47 CSVs, 2000-2023) ──→ 105,186 rows        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  SENTIMENT LAYER                                │
│  FinBERT (ProsusAI) ──→ pos/neg/neu scores per headline        │
│  7-day rolling mean + 1-day lag (no lookahead bias)            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   WIFCM LAYER  (Novel Contribution)            │
│                                                                 │
│  rᵢⱼ = d²ᵢⱼ / Σd²ᵢₖ  (relative distance)                     │
│                                                                 │
│  μ = ((1 - r^α) / (1 - (β·r)^α))^(1/α)   ← bullish degree    │
│  ν = r^α                                   ← bearish degree    │
│  π = 1 - μ - ν                             ← uncertainty       │
│                                                                 │
│  signal = (μ - ν) × (1 - π)                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  FORECASTING LAYER                              │
│  σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ + κ·sent²ₜ₋₁  (GARCH-X)      │
│  κ > 0 → negative sentiment increases volatility forecast      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  PORTFOLIO LAYER                                │
│  μ > 0.6  → Full exposure (100%)                               │
│  π > 0.4  → Half exposure (50%)  ← hold cash when uncertain   │
│  ν > 0.6  → Exit (0%)                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                   MLOPS LAYER                                   │
│  MLflow ──→ experiment tracking, model registry                │
│  Evidently AI ──→ weekly drift detection on μ/ν/π              │
│  GitHub Actions ──→ weekly CI/CD, auto-retrain on drift >15%  │
│  Streamlit Cloud ──→ live 5-page dashboard                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Research Foundation

This project extends **"A Dual-Pathway Tunable Modified Intuitionistic Fuzzy C-Means Framework with Dempster-Shafer Fusion for Melanoma Lesion Segmentation"** (VIT Chennai, 2025) into quantitative finance.

| | Research Paper | This Project |
|---|---|---|
| **Input** | LAB + Grayscale pixels | GDELT news headlines |
| **Algorithm** | WIFCM clustering | WIFCM sentiment scoring |
| **μ means** | Lesion membership | Bullish confidence |
| **ν means** | Non-membership | Bearish confidence |
| **π means** | Boundary uncertainty | Market uncertainty |
| **Fusion** | Dempster-Shafer | GARCH-X variance eq. |
| **Output** | Binary lesion mask | Graded portfolio exposure |
| **Metric** | Jaccard 0.9167 | Sharpe +39.78% |

> **Uniqueness:** No other published work applies WIFCM with Dempster-Shafer fusion to financial sentiment. This is the first known extension of this framework from medical imaging to quantitative finance.

---

## ⚡ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **NLP** | FinBERT (ProsusAI) | Financial sentiment classification |
| **Fuzzy** | WIFCM (custom) | μ/ν/π degree computation |
| **Volatility** | GARCH-X (arch + scipy) | Sentiment-augmented forecasting |
| **News** | GDELT API | Free historical news, no limits |
| **Experiment** | MLflow | Parameter + metric tracking |
| **Drift** | Evidently AI | Distribution monitoring |
| **CI/CD** | GitHub Actions | Weekly automated pipeline |
| **Dashboard** | Streamlit + Plotly | 5-page interactive interface |
| **Deployment** | Streamlit Cloud | Live public URL |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/kiruthikaJayaramanOfficial/Fuzzy-GARCH-FinBERT.git
cd Fuzzy-GARCH-FinBERT

# Setup
python3 -m venv fuzzy_env
source fuzzy_env/bin/activate
pip install -r requirements.txt

# Add your NewsAPI key (optional — GDELT works without any key)
echo "NEWSAPI_KEY=your_key_here" > .env

# Run full pipeline
python3 src/build_from_proven.py

# Launch dashboard
python3 -m streamlit run apps/streamlit_app/app.py
```

---

## 📁 Project Structure

```
Fuzzy-GARCH-FinBERT/
├── apps/
│   └── streamlit_app/
│       └── app.py              # 5-page dashboard
├── src/
│   ├── data_collection.py      # NIFTY-50 price download
│   ├── news_collection.py      # GDELT news fetching
│   ├── merge_data.py           # Price + news alignment
│   ├── sentiment.py            # FinBERT scoring
│   ├── fuzzy_index.py          # WIFCM μ/ν/π computation
│   ├── garch_model.py          # GARCH + GARCH-X fitting
│   ├── portfolio.py            # Fuzzy exposure strategy
│   ├── drift_detector.py       # Evidently drift detection
│   ├── build_from_proven.py    # Combined pipeline
│   └── stock_loader.py         # 47-stock loader
├── data/
│   ├── stocks/                 # 47 NIFTY stock CSVs
│   ├── fuzzy_index.csv         # μ/ν/π time series
│   ├── portfolio_results.csv   # Strategy returns
│   └── forecasts/              # Model comparison
├── eval/
│   ├── drift_report.html       # Evidently HTML report
│   └── drift_summary.json      # Drift trigger status
├── .github/
│   └── workflows/
│       └── retrain.yml         # Weekly CI/CD pipeline
└── requirements.txt
```

---

## 🤖 MLOps Pipeline

```
Every Monday 6am UTC (GitHub Actions):
│
├── 1. Fetch latest NIFTY prices (yfinance)
├── 2. Fetch latest GDELT headlines
├── 3. Score sentiment (FinBERT)
├── 4. Compute WIFCM μ/ν/π
├── 5. Detect drift (Evidently AI)
│       ├── drift > 15% → retrain GARCH-X → log to MLflow
│       └── drift ≤ 15% → skip retrain
├── 6. Push updated data to GitHub
└── 7. Streamlit Cloud auto-deploys new data
```

---

## 📈 Portfolio Optimizer (Page 5)

Select a real historical market event → WIFCM scores all 20 stocks using actual price data → shows optimized allocation vs equal weight:

| Event | WIFCM Decision | Real Outcome |
|-------|---------------|--------------|
| 🦠 COVID Crash | Avoid banking/metals → hold Pharma | Pharma +30%, Banks -52% ✅ |
| 💰 Budget 2021 | Overweight Infra/Banking | Infra rally confirmed ✅ |
| 📈 RBI Hike | Reduce NBFC/Banking | NBFC fell -63% ✅ |
| ⚔️ Russia-Ukraine | Energy positive, reduce Infra | Energy outperformed ✅ |

---

## 🎓 Skills Demonstrated

```
Machine Learning    → FinBERT NLP, GARCH-X time series, fuzzy clustering
Research Transfer   → Novel application of academic WIFCM to finance
MLOps              → MLflow + Evidently + GitHub Actions + Streamlit Cloud
Data Engineering   → GDELT API, yfinance, 105K+ rows multi-source merge
Quantitative Finance → Sharpe ratio, volatility forecasting, portfolio optimization
Software Engineering → Modular Python, CI/CD, drift detection, auto-retrain
```

---

## 📬 Contact

**Kiruthika Jayaraman** · VIT Chennai · Mathematics & Data Science

[![GitHub](https://img.shields.io/badge/GitHub-kiruthikaJayaramanOfficial-black?style=flat-square&logo=github)](https://github.com/kiruthikaJayaramanOfficial)

---

<div align="center">
<i>Built with research-grade methodology · Production-grade MLOps · Real market data</i>
</div>
