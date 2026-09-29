from __future__ import annotations

from typing import Tuple
import os, subprocess
from .default import DATA_DIR, DEVICE_NAME

from filelock import FileLock
import urllib.request
from functools import lru_cache
import torch

import logging 
logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# dtype detection and management
# -----------------------------------------------------------

@lru_cache() 
def get_rank():
    return int(os.environ.get("RANK", 0))

@lru_cache() 
def is_rank0():
    return get_rank() == 0

def format_value(value):
    """Format a value for display, adding commas to numbers and limiting floats to 4 decimal places."""
    if isinstance(value, bool):
        return "True" if value else "False"
    elif isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float) and (abs(value) < 1e-3 or abs(value) >= 1e7):
        return f"{value:.2e}"
    elif isinstance(value, float) and abs(value) < 1e7:
        return f"{value:,.2f}"
    else:
        return str(value)

def print0(*values, **kwargs):
    """Print message only if on global rank 0"""
    if is_rank0():
        print(*values, **kwargs)

def print0_dict(title, info):
    lines = [f"{title}:"]
    for k, v in info.items():
        lines.append(f"\t{k:<25}: {format_value(v):<60}")
    print0("\n".join(lines), end="\n\n")

def get_banner(to_print: bool = False) -> str:
    """Banner made with https://manytools.org/hacker-tools/ascii-banner/"""
    banner = r"""
      ___           ___       ___       ___           ___     
     /\  \         /\__\     /\__\     /\  \         /\  \    
    /::\  \       /:/  /    /:/  /    /::\  \       /::\  \   
   /:/\:\  \     /:/  /    /:/  /    /:/\:\  \     /:/\:\  \  
  /::\~\:\  \   /:/  /    /:/  /    /::\~\:\  \   /::\~\:\__\ 
 /:/\:\ \:\__\ /:/__/    /:/__/    /:/\:\ \:\__\ /:/\:\ \:|__|
 \/_|::\/:/  / \:\  \    \:\  \    \/__\:\/:/  / \:\~\:\/:/  /
    |:|::/  /   \:\  \    \:\  \        \::/  /   \:\ \::/  / 
    |:|\/__/     \:\  \    \:\  \       /:/  /     \:\/:/  /  
    |:|  |        \:\__\    \:\__\     /:/  /       \::/__/   
     \|__|         \/__/     \/__/     \/__/         ~~                                                                       
"""
    if to_print:
        print0(banner)
    return banner

def run_command(cmd):
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # Return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except:
        return None

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = DATA_DIR / "core"
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def detect_device_type() -> str:
    return DEVICE_NAME

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

@lru_cache()
def detect_compute_dtype() -> Tuple[torch.dtype, str]:
    env = os.environ.get("RLLAB_DTYPE", None)
    if env is not None:
        return _DTYPE_MAP[env], f"set via RLLAB_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via GPTLAB_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"

# from: https://github.com/karpathy/nanochat/blob/1076f97/nanochat/common.py
# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        (["a10g"], 70e12),
        (["a10"], 125e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    from .logging import log0
    log0(f"Peak flops undefined for: {device_name}, MFU will show as 0%", level="warning", logger=logger)
    return float('inf')


