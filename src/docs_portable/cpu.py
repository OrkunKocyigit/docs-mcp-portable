"""How the model servers share the CPU.

Measured with bge-m3 int8 on 16 physical cores: two onnxruntime processes using all cores with idle spinning on
ran 2.6x slower than one after the other; turning spinning off fixed that at no cost to a single process, and
half the cores each was best when both were busy (one process alone loses only ~5% on half the cores).
The reranker keeps all cores: its short calls lost much more (10 x 512-token pairs: 2.0 s -> 3.3 s on half).
"""

from __future__ import annotations

import ctypes
import os
import sys

from .config import Config

# GetLogicalProcessorInformationEx: RelationProcessorCore records, each starting with Relationship and Size (DWORDs).
_RELATION_PROCESSOR_CORE = 0


def physical_cores() -> int:
    """Physical cores (what onnxruntime and PyTorch use by default); logical processors if that fails."""
    logical = os.cpu_count() or 1
    if sys.platform != "win32":
        return logical
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, None, ctypes.byref(size))
        if not size.value:
            return logical
        buf = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, buf, ctypes.byref(size)):
            return logical
        count = offset = 0
        while offset + 8 <= size.value:
            record = ctypes.c_ulong.from_buffer(buf, offset + 4).value
            if record <= 0:
                break
            count += 1
            offset += record
        return min(max(count, 1), logical)
    except (OSError, AttributeError, ValueError):
        return logical


def shares_cpu(cfg: Config) -> bool:
    """True when docs embedding/rerank and bge-code-v1 all run on the CPU and can be busy at the same time.

    A docs search embeds the query and then reranks, one after the other, so those two never compete; the overlap
    is code indexing (PyTorch) running during a docs scrape or search (onnxruntime).
    """
    return cfg.code_search.enabled and cfg.gpu.provider == "none"


def thread_plan(cfg: Config, cores: int | None = None) -> dict[str, int]:
    """Intra-op threads per server ("embedding", "rerank", "code"); 0 = the runtime default (all physical cores).

    A `threads` value set in config.toml always wins.
    """
    plan = {"embedding": 0, "rerank": 0, "code": 0}
    if shares_cpu(cfg):
        cores = cores or physical_cores()
        code = max(1, cores // 2)
        docs = max(1, cores - code) if cores > 1 else 1
        plan = {"embedding": docs, "rerank": 0, "code": code}
    for name, configured in (("embedding", cfg.embedding.threads), ("rerank", cfg.rerank.threads),
                             ("code", cfg.code_search.threads)):
        if configured > 0:
            plan[name] = configured
    return plan
