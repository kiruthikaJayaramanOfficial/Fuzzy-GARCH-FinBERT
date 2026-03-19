# 📈 Fuzzy-GARCH-FinBERT Volatility Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.1-0194E2?style=for-the-badge)
![Evidently](https://img.shields.io/badge/Evidently_AI-0.7-7C3AED?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![FinBERT](https://img.shields.io/badge/FinBERT-NLP-orange?style=for-the-badge)

**NIFTY-50 sentiment-volatility forecasting** using WIFCM μ/ν/π fuzzy degrees
replacing hard thresholds · Full production MLOps pipeline

### 🚀 [Live Dashboard](https://your-app.streamlit.app) ← *Replace with your deployed URL*

</div>

---

## 🎯 What This Project Does

Traditional quant systems treat news sentiment as binary — *bullish = buy, bearish = sell*. This ignores a critical third state: **uncertainty**. What if the model genuinely doesn't know which way the market will move?

This project introduces **Weighted Intuitionistic Fuzzy C-Means (WIFCM)** to financial markets — converting FinBERT sentiment scores into three graded degrees:

| Degree | Symbol | Finance Meaning | Portfolio Action |
|--------|--------|-----------------|-----------------|
| Membership | **μ** | Bullish confidence | Buy proportionally |
| Non-membership | **ν** | Bearish confidence | Reduce exposure |
| Hesitancy | **π** | Market uncertainty | Hold cash |

**Result:** A self-monitoring, auto-retraining system that beats buy-and-hold by **+39.78% Sharpe** and improves GARCH volatility forecasting by **+11.37% MAE**.

---

## 📸 Dashboard — 5 Interactive Pages

### Page 1 — Sentiment Intelligence
*FinBERT → WIFCM μ/ν/π · 7-day rolled sentiment · Regime distribution donut · Lookback window slider*

![Sentiment Intelligence](docs/screenshots/main1.png)

---

### Page 2 — Volatility Forecast
*GARCH(1,1) Baseline vs GARCH-X+WIFCM · MAE comparison bars · Realized vol overlay · Improvement gauge*

![Volatility Forecast](docs/screenshots/main2.png)

---

### Page 3 — Portfolio Optimizer
*Select real market event → WIFCM scores 20 NIFTY stocks → Top N picks with ₹ allocation · Signal vs actual return scatter*

![Portfolio Optimizer](docs/screenshots/main3.png)

---

## 📊 Proven Results

### Volatility Forecasting · 366 test days (Jan 2022 – Jun 2023)

| Model | MAE | RMSE | Improvement |
|-------|-----|------|-------------|
| GARCH(1,1) Baseline | 1.1244 | 1.2131 | — |
| **GARCH-X + WIFCM** | **0.9965** | **1.0720** | **MAE ↓11.37% · RMSE ↓11.63%** |

### Portfolio Performance

| Strategy | Sharpe Ratio | Total Return |
|----------|-------------|--------------|
| Buy & Hold (baseline) | 0.7536 | 21.31% |
| **Fuzzy-WIFCM** | **1.0534** | **29.03%** |
| **Improvement** | **+39.78%** | **+7.72pp** |

### Real Market Event Validation (Portfolio Optimizer)

| Event | WIFCM Decision | Actual Outcome |
|-------|---------------|----------------|
| 🦠 COVID Crash Mar 2020 | Avoid Banking/Metals → Hold Pharma | Pharma +30%, Banks −52% ✅ |
| 💰 Union Budget Feb 2021 | Overweight Infra/Banking | Infra rally confirmed ✅ |
| 📈 RBI Rate Hike May 2022 | Reduce NBFC/Banking exposure | NBFC fell −63% ✅ |
| ⚔️ Russia-Ukraine Feb 2022 | Energy positive, reduce Infra | Energy outperformed ✅ |
| 🏦 Adani Crisis Jan 2023 | Avoid Infra/Banking | Infra crashed −90% ✅ |

---

## 🆚 WIFCM vs Traditional Approaches

```
┌──────────────────────────────────┬───────────────────────────────────────┐
│  TRADITIONAL HARD THRESHOLD      │  WIFCM FUZZY APPROACH                 │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Sentiment > 0  → BUY 100%        │ μ=0.72, ν=0.18, π=0.10               │
│ Sentiment < 0  → SELL 0%         │ Signal = (0.72−0.18)×(1−0.10) = 0.49 │
│ No middle ground                 │ Exposure = 48.6% (graded)             │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Binary: Bullish OR Bearish       │ Three states: Bull + Bear + Uncertain  │
│ Ignores uncertainty              │ π explicitly models "we don't know"    │
│ Overreacts to weak signals       │ Dampened by hesitancy degree           │
│ No confidence weighting          │ Allocation proportional to conviction  │
├──────────────────────────────────┼───────────────────────────────────────┤
│ Sharpe: 0.7536                   │ Sharpe: 1.0534  (+39.78%)             │
└──────────────────────────────────┴───────────────────────────────────────┘

Real Example — Russia-Ukraine Feb 2022:
  Hard threshold : "bearish → full exit (0%)"
  WIFCM          : π=0.65 (high uncertainty) → 50% exposure
  Result         : Captured 50% of partial recovery → better risk-adjusted return
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                          │
│  GDELT API (free, no key, back to 2000) → 5,731 real headlines      │
│  47 NIFTY-50 stock CSVs → 105,186 rows (2000–2023)                  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│  SENTIMENT LAYER                                                     │
│  FinBERT (ProsusAI, 438MB) → pos/neg/neu confidence per headline    │
│  7-day rolling mean + 1-day lag  ← eliminates lookahead bias        │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│  WIFCM LAYER  ← Novel Contribution from Research Paper              │
│                                                                      │
│  rᵢⱼ = d²ᵢⱼ / Σd²ᵢₖ              relative distance normalisation   │
│  μ = ((1−rᵅ) / (1−(β·r)ᵅ))^(1/α) bullish membership degree        │
│  ν = rᵅ                           bearish non-membership degree     │
│  π = 1 − μ − ν                    hesitancy / uncertainty degree    │
│                                                                      │
│  signal = (μ − ν) × (1 − π)      dampened fuzzy signal             │
│  Tunable: α=1.5 · β=0.5 · m=2.0                                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│  GARCH-X FORECASTING LAYER                                          │
│  σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁ + κ·sent²ₜ₋₁                       │
│  κ > 0  → negative sentiment increases volatility forecast          │
│  Fitted via scipy.optimize (manual implementation, arch-independent) │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│  PORTFOLIO LAYER                                                     │
│  μ > 0.6  → Full exposure  100%   high bullish conviction           │
│  π > 0.4  → Half exposure   50%   hold cash — uncertain             │
│  ν > 0.6  → Exit market      0%   high bearish conviction           │
│  else     → Graded: (1 − π) exposure                                │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│  MLOPS LAYER                                                         │
│  MLflow          experiment tracking, parameter + metric registry   │
│  Evidently AI    weekly drift detection on μ/ν/π distributions      │
│  GitHub Actions  weekly CI/CD, auto-retrain when drift > 15%        │
│  Streamlit Cloud live 5-page public dashboard                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 MLOps Stack

### MLflow — Experiment Tracking & Model Registry

Every pipeline run automatically logs parameters, metrics, and source for full reproducibility and comparison.

![MLflow Experiments List](docs/screenshots/mlflow_main.png)

![MLflow Run Detail — Parameters & Metrics](docs/screenshots/mlflow_metric.png)

![MLflow Metric Charts](docs/screenshots/mlflow_metric_pic.png)

**Every run logs:**
```
Parameters : alpha=1.5, beta=0.5, news_source=GDELT, data_rows=366
Metrics    : MAE_baseline=1.1244  MAE_garchx=0.9965
             MAE_improvement=11.37%  RMSE_improvement=11.63%
             Sharpe_baseline=0.7536  Sharpe_fuzzy=1.0534
             Sharpe_improvement=39.78%
```

**Compare runs** side-by-side to find optimal α/β combination:

![MLflow Run Comparison — α vs Sharpe](docs/screenshots/compare_mlflow.png)

---

### Evidently AI — Drift Detection

Monitors weekly μ/ν/π distributions. Detects when market regime shifts require model retraining.

![Evidently Drift Report](docs/screenshots/drift.png)

**What this report proves:**
- **π (hesitancy) DETECTED as drifted** between Jan–Jun 2022 (bear: RBI hikes + Russia-Ukraine) and Jul–Dec 2022 (bull: recovery period)
- **Jensen-Shannon distance** quantifies how much each distribution shifted
- When `share_of_drifted_columns > 0.15` → `trigger_retrain = true` → GitHub Actions reruns pipeline automatically

**Why π drifted and not μ/ν:** Uncertainty was HIGH during the crisis, LOW during recovery. The hesitancy degree uniquely captured this regime change — proving WIFCM's three-state representation is richer than binary sentiment.

---

### GitHub Actions — Weekly CI/CD Pipeline

![GitHub Actions — Workflow Run](docs/screenshots/action.png)

```yaml
# Runs every Monday 6am UTC + on manual trigger
Every Monday:
  1. Checkout repo
  2. Fetch latest NIFTY prices (yfinance)
  3. Run drift detection (Evidently AI)
  4. Check drift_summary.json
     ├── trigger_retrain=true  → rerun garch_model.py → log to MLflow
     └── trigger_retrain=false → skip retrain, dashboard stays current
  5. Commit updated eval/ + data/ results
  6. Streamlit Cloud auto-deploys on push
```

**Zero manual intervention.** The system monitors, decides, and retrains itself.

---

## ⚡ Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **NLP** | FinBERT (ProsusAI) | transformers 4.57 | Financial headline sentiment scoring |
| **Fuzzy Math** | WIFCM (custom impl.) | — | μ/ν/π membership degree computation |
| **Volatility** | GARCH-X (arch + scipy) | arch 8.0 | Sentiment-augmented variance forecasting |
| **News Data** | GDELT API | — | Free, no key, historical back to 2000 |
| **Market Data** | yfinance | 1.2 | NIFTY-50 daily OHLCV prices |
| **Experiment** | MLflow | 3.1 | Parameter logging, metric tracking, registry |
| **Drift** | Evidently AI | 0.7 | Weekly distribution monitoring |
| **CI/CD** | GitHub Actions | — | Automated weekly retraining pipeline |
| **Dashboard** | Streamlit + Plotly | 1.41 | 5-page interactive dark-theme UI |
| **Deployment** | Streamlit Cloud | — | Live public URL, auto-deploys on push |

---

## 🔬 Research Foundation

Extends **"A Dual-Pathway Tunable Modified Intuitionistic Fuzzy C-Means Framework with Dempster-Shafer Fusion for Melanoma Lesion Segmentation"** *(VIT Chennai, 2025)* into quantitative finance.

| | Research Paper (Medical Imaging) | This Project (Finance) |
|---|---|---|
| **Input** | LAB + Grayscale dermoscopic pixels | GDELT financial news headlines |
| **μ (membership)** | Lesion pixel belonging | Bullish market confidence |
| **ν (non-membership)** | Background belonging | Bearish market confidence |
| **π (hesitancy)** | Boundary uncertainty | Market regime uncertainty |
| **α parameter** | Cluster transition sharpness | Sentiment sharpness control |
| **β parameter** | Membership asymmetric scaling | Sentiment scaling factor |
| **Fusion method** | Dempster-Shafer evidence theory | GARCH-X variance equation (κ·sent²) |
| **Output** | Binary lesion segmentation mask | Graded portfolio exposure [0, 1] |
| **Best metric** | Jaccard Index 0.9167, Dice 0.9561 | Sharpe +39.78%, MAE −11.37% |

> **Why this is unique:** No published work applies WIFCM with tunable dual parameters (α, β) to financial sentiment. This is the first known transfer of this medical imaging framework to quantitative finance — your research paper directly enables a new application domain.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/kiruthikaJayaramanOfficial/Fuzzy-GARCH-FinBERT.git
cd Fuzzy-GARCH-FinBERT

# Create virtual environment
python3 -m venv fuzzy_env
source fuzzy_env/bin/activate    # Mac/Linux
pip install -r requirements.txt

# Run full pipeline (uses proven pre-built data)
python3 src/build_from_proven.py

# Launch the 5-page dashboard
python3 -m streamlit run apps/streamlit_app/app.py
# Open: http://localhost:8501

# View MLflow experiment tracking
mlflow ui --host 0.0.0.0 --port 5001
# Open: http://localhost:5001

# View Evidently drift report
open eval/drift_report.html
```

---

## 📁 Project Structure

```
Fuzzy-GARCH-FinBERT/
├── apps/streamlit_app/
│   └── app.py                      # 5-page interactive dashboard
├── src/
│   ├── data_collection.py          # NIFTY-50 price download
│   ├── news_collection.py          # GDELT historical news fetching
│   ├── merge_data.py               # Price + news alignment
│   ├── sentiment.py                # FinBERT batch scoring
│   ├── fuzzy_index.py              # WIFCM μ/ν/π computation
│   ├── garch_model.py              # GARCH + GARCH-X (MLflow logging)
│   ├── portfolio.py                # Fuzzy exposure strategy
│   ├── drift_detector.py           # Evidently drift detection
│   ├── build_from_proven.py        # Full combined pipeline runner
│   └── stock_loader.py             # 47-stock NIFTY CSV loader
├── data/
│   ├── stocks/                     # 47 individual stock CSVs (105K rows)
│   ├── fuzzy_index.csv             # μ/ν/π time series (366 days)
│   ├── portfolio_results.csv       # Daily strategy returns
│   └── forecasts/model_comparison.csv
├── eval/
│   ├── drift_report.html           # Evidently interactive HTML report
│   └── drift_summary.json          # Drift trigger: {trigger_retrain: bool}
├── docs/screenshots/               # All dashboard + MLflow screenshots
├── .github/workflows/
│   └── retrain.yml                 # Weekly CI/CD pipeline
└── requirements.txt
```

---

## 🎓 Skills Demonstrated

| Skill Area | What Was Built |
|-----------|---------------|
| **NLP & Deep Learning** | FinBERT financial text classification, 1476-row batch inference |
| **Fuzzy Mathematics** | Novel WIFCM membership function with tunable α/β parameters |
| **Time Series** | GARCH-X volatility modelling, 366-day rolling 1-step-ahead forecast |
| **MLOps** | MLflow experiment tracking · Evidently drift monitoring · GitHub Actions CI/CD |
| **Quantitative Finance** | Sharpe ratio, portfolio optimisation, risk-adjusted exposure sizing |
| **Data Engineering** | GDELT + yfinance multi-source pipeline, 105K+ row stock dataset |
| **Software Engineering** | Modular Python, automated retraining, Streamlit Cloud deployment |
| **Research Transfer** | Academic WIFCM framework successfully extended to new domain |

---

## 📬 Contact

**Kiruthika Jayaraman** · VIT Chennai · Department of Mathematics, School of Advanced Sciences

[![GitHub](https://img.shields.io/badge/GitHub-kiruthikaJayaramanOfficial-181717?style=flat-square&logo=github)](https://github.com/kiruthikaJayaramanOfficial)

---

<div align="center">

*Research-grade methodology · Production MLOps · Real GDELT news · 105,186 stock data points*

**⭐ Star this repo if you found it useful!**

</div>
