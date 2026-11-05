"""Weight quantization for every model loader: "none", "int8" (default) or "nf4".

A level the model, runtime or device cannot do is not applied: the model loads at its normal precision and a
warning is printed when the server starts.

ONNX (bge-m3, bge-reranker):
  int8  CPU only (dynamic int8 MatMul/Gather). onnxruntime runs int8 on NVIDIA GPUs only through TensorRT.
        A shipped model_int8.onnx / model_quantized.onnx is used when the repository has one, otherwise
        `start.bat setup` builds model.int8.onnx from the fp32 model once.
  nf4   CPU and CUDA (MatMulBnb4, 4-bit NormalFloat weights, block 64), built once from the fp32 model.
        Embedding tables stay in full precision, so XLM-R models shrink less than with int8.
PyTorch (bge-code-v1, jina-reranker):
  int8  CUDA: bitsandbytes LLM.int8. CPU: torch dynamic int8 Linear layers.
  nf4   CUDA only (bitsandbytes 4-bit).
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

LEVELS = ("none", "int8", "nf4")
SHIPPED_INT8 = ("model_int8.onnx", "model_quantized.onnx")
NF4_BLOCK_SIZE = 64


def generated_name(level: str) -> str:
    return f"model.{level}.onnx"


@dataclass(frozen=True)
class OnnxPlan:
    """Which ONNX file to load, the precision that gives, and what (if anything) setup must build first."""

    file: str  # relative to the model directory, e.g. "onnx/model.int8.onnx"
    precision: str  # "fp32", "fp16", "int8" or "nf4", for logs and /health
    build_from: str | None = None  # fp32 source to quantize when `file` is missing
    warning: str | None = None

    @property
    def build_level(self) -> str | None:
        return self.precision if self.build_from else None


def onnx_plan(level: str, provider: str, *, fp32: str, fp16: str | None = None,
              shipped: tuple[str, ...] = ()) -> OnnxPlan:
    """Plan for an ONNX model.

    provider: [gpu] provider ("none", "cuda", "directml"). fp32 / fp16: the full-precision files (fp16 is the
    unquantized choice on a GPU when the repository has one). shipped: int8 files the repository provides.
    """
    folder = fp32.rsplit("/", 1)[0] + "/" if "/" in fp32 else ""
    on_gpu = provider != "none"
    normal = OnnxPlan(fp16, "fp16") if on_gpu and fp16 else OnnxPlan(fp32, "fp32")

    def unsupported(reason: str) -> OnnxPlan:
        return OnnxPlan(normal.file, normal.precision,
                        warning=f"quantization = \"{level}\" is not supported {reason}; loading {normal.precision}.")

    if level == "none":
        return normal
    if level == "int8":
        if on_gpu:
            return unsupported(f"on {provider} for ONNX models (onnxruntime runs int8 on GPUs only via TensorRT)")
        if shipped:
            return OnnxPlan(shipped[0], "int8")
        return OnnxPlan(folder + generated_name("int8"), "int8", build_from=fp32)
    if level == "nf4":
        if provider == "directml":
            return unsupported("on directml (onnxruntime has 4-bit kernels for the CPU and CUDA only)")
        return OnnxPlan(folder + generated_name("nf4"), "nf4", build_from=fp32)
    raise ValueError(f"unknown quantization level {level!r}")


def local_onnx_files(model_dir: Path) -> set[str]:
    """ONNX files present under <model_dir>/onnx, as repository-relative paths."""
    folder = model_dir / "onnx"
    return {f"onnx/{p.name}" for p in folder.glob("*.onnx")} if folder.is_dir() else set()


def embedding_plan(level: str, provider: str, available: set[str]) -> OnnxPlan:
    """BAAI/bge-m3 style repository: onnx/model.onnx (+ model.onnx_data), maybe a shipped int8 file."""
    shipped = tuple(f"onnx/{n}" for n in SHIPPED_INT8 if f"onnx/{n}" in available)
    return onnx_plan(level, provider, fp32="onnx/model.onnx", shipped=shipped)


def rerank_plan(level: str, provider: str, onnx_file: str, available: set[str]) -> OnnxPlan:
    """onnx-community reranker repository: model.onnx, model_fp16.onnx, model_int8.onnx, ..."""
    if onnx_file != "auto":
        warning = None if level == "none" else (
            f"[rerank] bge_onnx_file = \"{onnx_file}\" is set, so quantization = \"{level}\" is not applied.")
        return OnnxPlan(onnx_file, "as configured", warning=warning)
    fp16 = "onnx/model_fp16.onnx" if "onnx/model_fp16.onnx" in available else None
    shipped = tuple(f"onnx/{n}" for n in SHIPPED_INT8 if f"onnx/{n}" in available)
    return onnx_plan(level, provider, fp32="onnx/model.onnx", fp16=fp16, shipped=shipped)


def download_patterns(plan: OnnxPlan, available: set[str]) -> list[str]:
    """Repository files a plan needs: the file itself (unless setup builds it) and the fp32 source with its data."""
    files = [] if plan.build_from else [plan.file]
    if plan.build_from:
        files.append(plan.build_from)
    for name in list(files):
        if f"{name}_data" in available:
            files.append(f"{name}_data")
    return files


def build_onnx(source: Path, target: Path, level: str) -> None:
    """Quantize source (fp32, external data allowed) into one self-contained file; atomic."""
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    try:
        if level == "int8":
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(source, tmp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
        elif level == "nf4":
            import onnx
            from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer

            quantizer = MatMulBnb4Quantizer(onnx.load(str(source)), MatMulBnb4Quantizer.NF4, NF4_BLOCK_SIZE)
            quantizer.process()
            quantizer.model.save_model_to_file(str(tmp), use_external_data_format=False)
        else:
            raise ValueError(f"cannot build an ONNX {level!r} model")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def resolve_onnx(model_dir: Path, plan: OnnxPlan, log: logging.Logger) -> tuple[Path, str]:
    """File to load at run time. Warns (and falls back to fp32) if setup has not built the planned file."""
    if plan.warning:
        log.warning(plan.warning)
    path = model_dir / plan.file
    if path.exists():
        return path, plan.precision
    if plan.build_from and (model_dir / plan.build_from).exists():
        log.warning("%s has not been built yet (run `start.bat setup`); loading fp32.", plan.file)
        return model_dir / plan.build_from, "fp32"
    return path, plan.precision  # caller reports the missing model


# ------------------------------------------------------------------ PyTorch


@dataclass(frozen=True)
class TorchPlan:
    level: str  # what will be attempted: "none", "int8" or "nf4"
    warning: str | None = None


def torch_plan(level: str, cuda: bool) -> TorchPlan:
    if level == "nf4" and not cuda:
        return TorchPlan("none", "quantization = \"nf4\" needs an NVIDIA GPU (bitsandbytes 4-bit); "
                                 "loading without quantization.")
    return TorchPlan(level)


def bnb_config(level: str, compute_dtype):
    """bitsandbytes config for a CUDA load, or None."""
    if level == "none":
        return None
    from transformers import BitsAndBytesConfig

    if level == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)


def cpu_int8(model):
    """Dynamic int8 for every nn.Linear on the CPU (weights int8, activations quantized per batch).

    torch.ao.quantization is deprecated in favour of torchao but still ships with torch 2.14; if a later torch
    drops it, load_torch catches the error and runs the model unquantized.
    """
    import warnings

    import torch
    from torch.ao.quantization import quantize_dynamic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)  # inplace: no 2x RAM


def load_torch(load, level: str, cuda: bool, compute_dtype, log: logging.Logger, name: str):
    """Load a transformers model with the requested level; falls back to an unquantized load with a warning.

    load(quantization_config) must return the model (moved to its device for unquantized/CPU loads).
    Returns (model, precision actually applied).
    """
    plan = torch_plan(level, cuda)
    if plan.warning:
        log.warning("%s: %s", name, plan.warning)
    if plan.level == "none":
        return load(None), "none"
    if cuda:
        try:
            return load(bnb_config(plan.level, compute_dtype)), plan.level
        except (ImportError, AttributeError, ValueError, RuntimeError, NotImplementedError, TypeError) as exc:
            log.warning("%s: quantization = \"%s\" is not supported for this model (%s); loading without "
                        "quantization.", name, plan.level, str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__)
            return load(None), "none"
    model = load(None)
    try:
        return cpu_int8(model), "int8"
    except (ImportError, AttributeError, RuntimeError, NotImplementedError, TypeError, AssertionError) as exc:
        log.warning("%s: int8 on the CPU is not supported for this model (%s); running without quantization.",
                    name, str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__)
        return model, "none"
