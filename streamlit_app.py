"""
Streamlit Dashboard – MAF/RealNVP + Flow Portfolio Allocation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Normalizing Flow", page_icon="🌀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("normalizing_flow_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        path = hf_hub_download(repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
                               repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def safe_pct(val):
    try:
        return f"{float(val)*100:.2f}%"
    except:
        return "N/A"

def display_maf_tab(model_data):
    """MAF/RealNVP forecast tab (unchanged from original)."""
    if not model_data:
        st.warning("No data available.")
        return
    top_picks = model_data.get('top_picks', {})
    universes = model_data.get('universes', {})
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    for subtab, key in zip(subtabs, keys):
        with subtab:
            top = top_picks.get(key, [])
            universe = universes.get(key, {})
            if top:
                p = top[0]
                st.markdown(f"""
                <div class="hero-card">
                    <div style="font-size: 1.2rem; opacity: 0.8;">🌀 MAF/RealNVP TOP PICK</div>
                    <div class="hero-ticker">{p['ticker']}</div>
                    <div>Expected Return: {safe_pct(p['expected_return'])}</div>
                    <div>Annualized Vol: {safe_pct(p.get('annualized_vol', 0))}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Top 3 Picks")
                rows = [{"Ticker": p['ticker'], "Exp Return": safe_pct(p['expected_return']),
                         "Vol": safe_pct(p.get('annualized_vol', 0))} for p in top]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("### All ETFs")
                all_rows = [{"Ticker": t, "Exp Return": safe_pct(d['expected_return']),
                             "Vol": safe_pct(d.get('annualized_vol', 0))} for t, d in universe.items()]
                df_all = pd.DataFrame(all_rows).sort_values("Exp Return", ascending=False)
                st.dataframe(df_all, use_container_width=True, hide_index=True)

def display_portfolio_tab(weights_dict):
    """Flow‑based portfolio allocation tab."""
    if not weights_dict:
        st.warning("No portfolio weights available.")
        return
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    for subtab, key in zip(subtabs, keys):
        with subtab:
            weights = weights_dict.get(key, [])
            if weights:
                df = pd.DataFrame(weights, columns=['Ticker', 'Weight'])
                df['Weight'] = df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                st.markdown("### Flow‑Optimized Portfolio (Top 5)")
                st.dataframe(df, use_container_width=True, hide_index=True)
                # Pie chart
                import plotly.graph_objects as go
                fig = go.Figure(go.Pie(labels=df['Ticker'], values=[float(w.strip('%'))/100 for w in df['Weight']], hole=0.4))
                fig.update_layout(title_text=f"{key} – Flow Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No weights for {key}.")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
    rf = data.get('risk_free_rate', 0.04)
    st.sidebar.markdown(f"**🏦 3‑Month T‑Bill (Rf):** {rf*100:.2f}%")
else:
    rf = 0.04

st.markdown('<div class="main-header">🌀 P2Quant Normalizing Flow</div>', unsafe_allow_html=True)
st.markdown('<div>RealNVP – Density Estimation & Flow‑Based Portfolio Optimization</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🌀 MAF/RealNVP", "📊 Flow Portfolio"])

with tab1:
    display_maf_tab(data.get('daily_trading'))

with tab2:
    display_portfolio_tab(data.get('daily_trading', {}).get('flow_portfolio_weights', {}))
