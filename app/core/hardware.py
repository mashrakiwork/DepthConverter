"""CUDA / hardware detection. Imports torch, so keep out of UI import paths."""

import torch


def resolve_device(choice: str = "auto") -> str:
    choice = (choice or "auto").lower()
    if choice == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_summary() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"CUDA ready: {name} ({total:.0f} GB VRAM)"
    return "No CUDA GPU detected - running on CPU (slow)"


def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info(0)
    return free / (1024 ** 3)


def suggest_batch_size(model_id: str, device: str) -> int:
    """Initial inference batch size from free VRAM; the engine halves it on OOM."""
    if device != "cuda":
        return 1
    lower = model_id.lower()
    if "large" in lower or "giant" in lower:
        per_sample = 1.2
    elif "base" in lower or "hybrid" in lower:
        per_sample = 0.7
    else:
        per_sample = 0.35
    usable = max(free_vram_gb() - 1.5, per_sample)  # keep headroom for the OS/display
    return int(max(1, min(16, usable // per_sample)))
