# P2-ETF-NORMALIZING-FLOW

**RealNVP Normalizing Flow – Invertible Density Estimation for ETF Returns**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-NORMALIZING-FLOW/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-NORMALIZING-FLOW/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--normalizing--flow--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-normalizing-flow-results)

## Overview

`P2-ETF-NORMALIZING-FLOW` learns the joint distribution of 23 ETF daily returns using a **RealNVP** invertible normalizing flow. The model is trained on the full 2008‑2026 dataset. After training, the flow is sampled to obtain marginal expected returns and volatilities for each ETF, which are then ranked per universe.

## Methodology

- **RealNVP**: 8 affine coupling layers with residual networks for the scale and translation functions.
- **Training**: maximum likelihood on 4,000+ daily returns, 300 epochs.
- **Inference**: sample 50,000 points from the learned distribution and compute per‑ETF statistics.

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
