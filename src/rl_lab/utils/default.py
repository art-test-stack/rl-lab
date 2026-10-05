from __future__ import annotations

import torch
import os
from pathlib import Path

from functools import lru_cache

# ----------- PROCESSOR -----------

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
if MPS_AVAILABLE:
    torch.mps.empty_cache()
    torch.mps.set_per_process_memory_fraction(0.)
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

NUM_THREADS = os.cpu_count()

# ------------- DATA -------------

IS_TIKTOKEN = False # TODO: parse as arg


@lru_cache()
def get_cache_dir() -> Path:
    cache_dir = os.environ.get("RLLAB_CACHE_DIR", None)
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    else:
        return Path.home() / ".cache" / "rl_lab"
cache_dir = os.environ.get("RLLAB_CACHE_DIR", None)

CACHE_DIR = get_cache_dir()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BOARD_DIR = CACHE_DIR / "runs"
DATA_DIR = CACHE_DIR / "data"
MIN_DOCUMENT_SIZE = 0
OUTPUT_FOLDER = CACHE_DIR / "output"
MODELS_FOLDER = CACHE_DIR / "models"
REPORTS_FOLDER = CACHE_DIR / "reports"
TOKENIZERS_FOLDER = CACHE_DIR / "tokenizers"
VOCAB_SIZE = 32_000
MAX_TOKEN_LENGTH = 32
