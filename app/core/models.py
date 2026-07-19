"""Depth model registry + local download handling (HuggingFace hub cache).

The UI combo is editable: any HuggingFace depth-estimation repo id can be
typed in addition to the presets below.
"""

MODELS: list[tuple[str, str]] = [
    ("Depth Anything V3 Small", "depth-anything/DA3-SMALL"),
    ("Depth Anything V3 Base", "depth-anything/DA3-BASE"),
    ("Depth Anything V3 Large", "depth-anything/DA3-LARGE"),
    ("Depth Anything V3 Mono Large (best for 2D->3D)", "depth-anything/DA3MONO-LARGE"),
    ("Depth Anything V2 Small (fastest)", "depth-anything/Depth-Anything-V2-Small-hf"),
    ("Depth Anything V2 Base", "depth-anything/Depth-Anything-V2-Base-hf"),
    ("Depth Anything V2 Large", "depth-anything/Depth-Anything-V2-Large-hf"),
    ("DPT Large (MiDaS 3.0)", "Intel/dpt-large"),
    ("DPT Hybrid (MiDaS)", "Intel/dpt-hybrid-midas"),
]


def is_da3(model_id: str) -> bool:
    """Depth Anything V3 repos need the dedicated depth_anything_3 package."""
    return "/da3" in model_id.lower()


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
