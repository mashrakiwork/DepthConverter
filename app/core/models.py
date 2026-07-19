"""Depth model registry + local download handling (HuggingFace hub cache).

The UI combo is editable: any HuggingFace depth-estimation repo id can be
typed in addition to the presets below.
"""

MODELS: list[tuple[str, str]] = [
    ("Depth Anything V2 Small (fastest)", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("Depth Anything V2 Base", "depth-anything/Depth-Anything-V2-Base-hf"),
    ("Depth Anything V2 Large (best quality)", "depth-anything/Depth-Anything-V2-Large-hf"),
    ("DPT Large (MiDaS 3.0)", "Intel/dpt-large"),
    ("DPT Hybrid (MiDaS)", "Intel/dpt-hybrid-midas"),
]


def ensure_downloaded(repo_id: str, log=print) -> str:
    """Make sure the model is in the local HF cache; download only if missing."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        log(f"Model '{repo_id}' not cached yet - downloading from Hugging Face "
            f"(one-time, this can take a while)...")
        path = snapshot_download(repo_id)
        log("Model download complete.")
        return path
