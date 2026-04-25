"""
Main training script for Normalizing Flow engine.
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

def run_flow():
    print(f"=== P2-ETF-NORMALIZING-FLOW Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

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
    samples_orig = scaler.inverse_transform(samples)  # back to original return scale

    # Compute per-ETF expected return (annualized)
    expected_returns = {}
    std_devs = {}
    for i, tkr in enumerate(tickers):
        daily_ret = samples_orig[:, i]
        expected_returns[tkr] = float(np.mean(daily_ret) * 252)  # annualized
        std_devs[tkr] = float(np.std(daily_ret) * np.sqrt(252))

    # Build per-universe results
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

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_flow()
