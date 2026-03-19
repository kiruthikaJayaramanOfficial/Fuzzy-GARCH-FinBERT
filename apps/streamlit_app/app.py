import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Fuzzy-GARCH-FinBERT",
    page_icon="📈",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    fuzzy   = pd.read_csv("data/fuzzy_index.csv",
                          index_col="Date", parse_dates=True)
    port    = pd.read_csv("data/portfolio_results.csv",
                          index_col="Date", parse_dates=True)
    summary = pd.read_csv("data/portfolio_summary.csv")
    compare = pd.read_csv("data/forecasts/model_comparison.csv")
    return fuzzy, port, summary, compare

fuzzy, port, summary, compare = load_data()

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", [
    "📰 Page 1 — Live Sentiment",
    "📉 Page 2 — Volatility Forecast",
    "💼 Page 3 — Portfolio Simulator"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Fuzzy-GARCH-FinBERT")
st.sidebar.markdown("**Index:** NIFTY-50")
st.sidebar.markdown("**WIFCM α=1.5, β=0.5**")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Live Sentiment Feed
# ══════════════════════════════════════════════════════════════
if page == "📰 Page 1 — Live Sentiment":
    st.title("📰 Live Sentiment Feed")
    st.markdown("FinBERT scores converted to WIFCM **μ/ν/π** fuzzy degrees")

    # Latest fuzzy values
    latest = fuzzy[["mu","nu","pi","fuzzy_sentiment"]].dropna().iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("μ Bullish",    f"{latest['mu']:.4f}")
    col2.metric("ν Bearish",    f"{latest['nu']:.4f}")
    col3.metric("π Uncertainty",f"{latest['pi']:.4f}")
    col4.metric("Fuzzy Signal", f"{latest['fuzzy_sentiment']:.4f}")

    st.markdown("---")

    # 7-day fuzzy trend
    st.subheader("7-Day Fuzzy Sentiment Trend")
    recent = fuzzy[["mu","nu","pi"]].dropna().tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["mu"],
                             name="μ Bullish", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["nu"],
                             name="ν Bearish", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["pi"],
                             name="π Uncertainty", line=dict(color="orange",
                             dash="dash")))
    fig.update_layout(height=400, xaxis_title="Date",
                      yaxis_title="Fuzzy Degree",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Volatility Forecast
# ══════════════════════════════════════════════════════════════
elif page == "📉 Page 2 — Volatility Forecast":
    st.title("📉 Volatility Forecast")
    st.markdown("GARCH model comparison: Baseline vs Hard-Threshold vs **Fuzzy-WIFCM**")

    # Model comparison bar chart
    st.subheader("Model MAE Comparison")
    fig2 = px.bar(compare, x="Model", y="MAE",
                  color="Model",
                  color_discrete_map={
                      "Baseline":       "#636EFA",
                      "Hard-Threshold": "#EF553B",
                      "Fuzzy-GARCH":    "#00CC96"
                  },
                  text="MAE")
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # NIFTY log returns
    st.subheader("NIFTY-50 Log Returns")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=fuzzy.index, y=fuzzy["log_return"],
        name="Log Return", line=dict(color="#636EFA", width=1)
    ))
    fig3.update_layout(height=350, xaxis_title="Date",
                       yaxis_title="Log Return")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — Portfolio Simulator
# ══════════════════════════════════════════════════════════════
elif page == "💼 Page 3 — Portfolio Simulator":
    st.title("💼 Portfolio Simulator")
    st.markdown("Fuzzy exposure strategy vs always-invested baseline")

    # Sharpe metrics
    col1, col2, col3 = st.columns(3)
    base_sharpe  = summary.loc[summary.Strategy=="Baseline",   "Sharpe"].values[0]
    fuzzy_sharpe = summary.loc[summary.Strategy=="Fuzzy-WIFCM","Sharpe"].values[0]
    improvement  = ((fuzzy_sharpe / base_sharpe) - 1) * 100 if base_sharpe != 0 else 0

    col1.metric("Baseline Sharpe",  f"{base_sharpe:.4f}")
    col2.metric("Fuzzy Sharpe",     f"{fuzzy_sharpe:.4f}")
    col3.metric("Improvement",      f"{improvement:.2f}%")

    st.markdown("---")

    # Cumulative returns chart
    st.subheader("Cumulative Returns Comparison")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=port.index, y=port["cumulative_baseline"],
        name="Baseline", line=dict(color="#EF553B", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=port.index, y=port["cumulative_fuzzy"],
        name="Fuzzy-WIFCM", line=dict(color="#00CC96", width=2)
    ))
    fig4.update_layout(height=400, xaxis_title="Date",
                       yaxis_title="Cumulative Log Return",
                       legend=dict(orientation="h"))
    st.plotly_chart(fig4, use_container_width=True)

    # Slider — threshold control
    st.markdown("---")
    st.subheader("🎛️ Adjust Fuzzy Threshold")
    threshold = st.slider("π uncertainty threshold (hold cash if π >)",
                          0.0, 1.0, 0.4, 0.05)
    st.info(f"Current threshold: **{threshold}** — "
            f"rows with π > {threshold} get 50% exposure")