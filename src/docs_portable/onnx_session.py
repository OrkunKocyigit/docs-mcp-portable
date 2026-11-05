"""onnxruntime session setup shared by the embedding and rerank servers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import GpuConfig


def session_options(gpu: GpuConfig, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        options.intra_op_num_threads = threads
    # Idle worker threads would otherwise spin after each run, starving the other model servers on the same
    # CPU (2.6x slower when two ran at once); measured no cost to a single server.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if gpu.provider == "directml":
        # Required by the DirectML execution provider.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def create_session(model_path: Path, gpu: GpuConfig, threads: int, label: str):
    import onnxruntime as ort

    if gpu.provider == "cuda" and hasattr(ort, "preload_dlls"):
        # CUDA/cuDNN DLLs come from the nvidia-* wheels installed with onnxruntime-gpu[cuda,cudnn].
        ort.preload_dlls()

    options = session_options(gpu, threads)

    available = set(ort.get_available_providers())
    providers = [p for p in gpu.onnx_providers() if (p[0] if isinstance(p, tuple) else p) in available]

    log = logging.getLogger(label)
    started = time.perf_counter()
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    active = session.get_providers()
    log.info("Loaded %s with %s in %.1fs (CPU threads: %s)", model_path.name, active, time.perf_counter() - started,
             threads if threads > 0 else "all physical cores")
    if gpu.enabled and active[0] == "CPUExecutionProvider":
        log.warning(
            "[gpu] provider = %r was requested but is not usable (available: %s); running on the CPU. "
            "Check the GPU driver, or run `start.bat setup` after changing [gpu].",
            gpu.provider, ", ".join(sorted(available)),
        )
    return session


def run_options(gpu: GpuConfig):
    """Per-run options; on CUDA, hand unused arena memory back after every run.

    The embedding and rerank servers share one GPU. Without this each keeps the peak memory of its
    largest request forever, and together they overflow into (much slower) shared system memory.
    """
    if gpu.provider != "cuda":
        return None
    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{gpu.device_id}")
    return options


def device_name(session) -> str:
    return {"CUDAExecutionProvider": "cuda", "DmlExecutionProvider": "directml"}.get(session.get_providers()[0], "cpu")
