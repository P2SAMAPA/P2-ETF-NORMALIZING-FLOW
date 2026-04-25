"""
Main training script – RealNVP + Flow‑based Portfolio Optimization.
"""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

import config
import data_manager
from flow_model import RealNVPFlow
import push_results

# ----------------------------------------------------------------------
def compute_flow_portfolio_weights(flow, scaler, returns, n_samples, risk_free_rate):
    """
    Compute optimal Sharpe‑ratio weights for each universe using the flow.
    Returns a dict: weights[universe] -> list of (ticker, weight)
    """
    # Sample from the flow
    samples = flow.sample(n_samples)                     # (num_samples, dim_all)
    samples_orig = scaler.inverse_transform(samples)

    tickers_all = returns.columns.tolist()

    weights_by_universe = {}
    for uni_name, uni_tickers in config.UNIVERSES.items():
        idx = [tickers_all.index(t) for t in uni_tickers if t in tickers_all]
        if len(idx) < 2:
            weights_by_universe[uni_name] = []
            continue

        uni_samples = samples_orig[:, idx]               # (n_samples, n_uni)
        mean_returns = np.mean(uni_samples, axis=0) * 252
        cov_mat = np.cov(uni_samples.T) * 252

        excess_returns = mean_returns - risk_free_rate
        try:
            inv_cov = np.linalg.inv(cov_mat)
            raw_weights = inv_cov @ excess_returns
            raw_weights = np.maximum(raw_weights, 0)       # long‑only
            raw_weights /= raw_weights.sum()
        except np.linalg.LinAlgError:
            raw_weights = np.ones(len(idx)) / len(idx)

        paired = list(zip(uni_tickers, raw_weights))
        paired.sort(key=lambda x: x[1], reverse=True)
        top5 = paired[:5]
        total = sum(w for _, w in top5)
        weights_by_universe[uni_name] = [(t, w / total) for t, w in top5]

    return weights_by_universe

# ----------------------------------------------------------------------
def run_flow():
    print(f"=== P2-ETF-NORMALIZING-FLOW Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]
    macro = data_manager.prepare_macro(df_master)          # for T‑bill

    # Train on combined universe (all 23 ETFs)
    tickers = config.ALL_TICKERS
    returns = data_manager.prepare_returns_matrix(df_master, tickers)
    if len(returns) < config.MIN_OBSERVATIONS:
        print("Insufficient data")
        return

    # Scale returns to zero mean unit variance
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns.values)

    flow = RealNVPFlow(
        dim=len(tickers),
        num_layers=config.NUM_COUPLING_LAYERS,
        hidden_features=config.HIDDEN_FEATURES,
        lr=config.LEARNING_RATE,
        wd=config.WEIGHT_DECAY,
        seed=config.RANDOM_SEED
    )

    print(f"Training RealNVP on {len(scaled)} samples, {len(tickers)} dimensions...")
    flow.fit(scaled, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

    # Sample from learned distribution
    print(f"Sampling {config.NUM_SAMPLES} points...")
    samples = flow.sample(config.NUM_SAMPLES)  # (num_samples, dim)
    samples_orig = scaler.inverse_transform(samples)

    # Extract latest 3M T‑bill
    tbill_series = macro['TBILL_3M'].dropna()
    risk_free_rate = tbill_series.iloc[-1] / 100.0 if len(tbill_series) > 0 else 0.04
    print(f"Risk‑free rate (3M T‑bill): {risk_free_rate*100:.2f}%")

    # Per‑ETF expected return and vol
    expected_returns = {}
    std_devs = {}
    for i, tkr in enumerate(tickers):
        daily_ret = samples_orig[:, i]
        expected_returns[tkr] = float(np.mean(daily_ret) * 252)
        std_devs[tkr] = float(np.std(daily_ret) * np.sqrt(252))

    # Build per‑universe results (MAF/RealNVP)
    all_results = {}
    top_picks = {}
    for uni_name, uni_tickers in config.UNIVERSES.items():
        universe_data = {}
        for tkr in uni_tickers:
            if tkr in expected_returns:
                universe_data[tkr] = {
                    "ticker": tkr,
                    "expected_return": expected_returns[tkr],
                    "annualized_vol": std_devs[tkr]
                }
        all_results[uni_name] = universe_data
        sorted_items = sorted(universe_data.items(), key=lambda x: x[1]["expected_return"], reverse=True)
        top_picks[uni_name] = [
            {"ticker": t, "expected_return": d["expected_return"], "annualized_vol": d["annualized_vol"]}
            for t, d in sorted_items[:3]
        ]

    # Flow‑based portfolio optimization
    flow_weights = compute_flow_portfolio_weights(flow, scaler, returns, config.NUM_SAMPLES, risk_free_rate)

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "risk_free_rate": risk_free_rate,
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks,
            "flow_portfolio_weights": flow_weights
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_flow()
