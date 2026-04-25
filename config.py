"""
Configuration for P2-ETF-NORMALIZING-FLOW engine.
"""

import os
from datetime import datetime

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-normalizing-flow-results"

FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))
UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# Flow parameters
NUM_COUPLING_LAYERS = 8
HIDDEN_FEATURES = 256
EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252
TRAIN_START = "2008-01-01"

# Sampling
NUM_SAMPLES = 50_000

TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
