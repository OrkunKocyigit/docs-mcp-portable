# docs-portable

One double-click (`start.bat`) gives you a **self-contained, portable** documentation server:

- [docs-mcp-server](https://github.com/arabold/docs-mcp-server) (MIT) – indexes documentation and serves it over MCP, a CLI and a web UI.
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT) - a multilingual embedding model, served locally through an
  **OpenAI-compatible `/v1/embeddings` endpoint** and wired into docs-mcp-server for vector search.
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (Apache-2.0) - a multilingual reranker
  that re-scores the best search candidates before docs-mcp-server returns them
  ([docs-mcp-server#472](https://github.com/arabold/docs-mcp-server/issues/472)). Optionally
  [jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) for **non-commercial** use.

No API keys, nothing leaves your machine.

The repository itself is tiny. On the first run `start.bat` downloads everything into `runtime\`; delete that folder to
reset, or copy the whole folder to another machine/USB stick (same OS/architecture) and keep going.

## Quick start

```bat
start.bat
```

First run (needs internet, roughly 3 GB, a few minutes):

| Step | What | Where |
|------|------|-------|
| 1 | Python **3.12** (official embeddable build, fetched with Windows' built-in `curl`/`tar`) | `runtime\python` |
| 2 | pip + dependencies (`fastapi`, `onnxruntime`, `tokenizers`, ...) via `bootstrap.py` | `runtime\python\Lib\site-packages` |
| 3 | Portable Node.js (checksum-verified) | `runtime\node` |
| 4 | `@arabold/docs-mcp-server` (npm) | `runtime\app` |
| 5 | Chromium for JS-rendered sites (Playwright) | `runtime\browsers` |
| 6 | BGE-M3 ONNX model (~2.3 GB, resumable) | `runtime\models` |
| 7 | bge-reranker-v2-m3 ONNX model (int8 ~570 MB on CPU, fp16 ~1.1 GB on GPU) + rerank patch for docs-mcp-server | `runtime\models`, `runtime\app` |
| 8 | Only with `[jev] enabled = true`: PyTorch CUDA extras, JEV-9B (~18 GB) and jev-mcp | `runtime\python`, `runtime\models`, `runtime\jev-mcp` |
| 9 | Only with `[code_search] enabled = true`: cocoindex-code, PyTorch and bge-code-v1 (~6.2 GB) | `runtime\python`, `runtime\models`, `runtime\data\code` |

Later runs start in seconds and work offline (except for scraping, of course).

Then open:

- Web UI: <http://127.0.0.1:6280>
- MCP endpoint: `http://127.0.0.1:6280/mcp`
- Embeddings (OpenAI API): `http://127.0.0.1:8081/v1` (model `bge-m3`)
- Reranking (Jina/Cohere-style): `http://127.0.0.1:8082/v1/rerank` (model `bge-reranker-v2-m3`)

MCP client entry:

```json
{ "mcpServers": { "docs-mcp-server": { "url": "http://127.0.0.1:6280/mcp" } } }
```

Stop with `Ctrl+C` (or just close the window – all child processes are killed with it).

## Commands

```bat
start.bat                    rem install what is missing, then run everything
start.bat setup              rem only download/install
start.bat embedding-server   rem only the embedding endpoint
start.bat rerank-server      rem only the rerank endpoint
start.bat jev-server         rem only the JEV-9B endpoint (needs DOCS_PORTABLE_JEV_TOKEN, 32+ chars)
start.bat code-embedding-server  rem only the bge-code-v1 endpoint ([code_search] enabled)
start.bat code-server        rem only the Code page + /mcp (needs the code embedding endpoint)
start.bat cli <args...>      rem docs-mcp-server CLI with the same portable data and embedding setup
```

CLI examples (run `start.bat` in another window first so embeddings are available):

```bat
start.bat cli scrape react https://react.dev/reference/react
start.bat cli search react "useEffect cleanup" --output yaml
start.bat cli --help
```

If a port from `config.toml` is already taken (e.g. you also run docs-mcp-server via `npx` on 6280), the launcher
refuses to start and tells you; change `port` in `config.toml`.

## How it works

```
start.bat ── curl/tar ──> runtime\python (3.12) ──> bootstrap.py (pip + deps) ──> python -m docs_portable
                                           │ 1. ensure Node, docs-mcp-server, Chromium, model
                                           │ 2. start embedding server (onnxruntime, 127.0.0.1:8081)
                                           │ 3. wait for /health, send a test embedding
                                           └ 4. start docs-mcp-server with
                                                OPENAI_API_BASE=http://127.0.0.1:8081/v1
                                                DOCS_MCP_EMBEDDING_MODEL=bge-m3
```

The embedding server runs the official ONNX export of BGE-M3 (dense CLS embedding, L2-normalised, 1024 dimensions,
up to 8192 tokens) on `onnxruntime` – no PyTorch, so the install stays small. It supports `float` and `base64`
encodings (the OpenAI Node SDK used by docs-mcp-server asks for base64).

docs-mcp-server runs with `APPDATA`/`LOCALAPPDATA` redirected into `runtime\data`, so its database
(`runtime\data\store\documents.db`), generated `config.yaml` and caches never touch your user profile.

## Reranking

docs-mcp-server ranks search candidates with Reciprocal Rank Fusion of vector and full-text search, then keeps the
top `limit`. It has no rerank step, so docs-portable adds one: after installing docs-mcp-server, setup patches
`DocumentStore.findByContent` in its `dist` bundle so the top `candidates` chunks (default 10 on CPU, 30 on GPU) are re-scored
by the local `/v1/rerank` endpoint first, before grouping and context assembly.

- The patch is an exact text replacement written for docs-mcp-server **3.4.1**. With any other version setup stops
  with an error instead of guessing; set `[rerank] enabled = false` or pin a supported version.
- If the rerank server is down, slow (`timeout_ms`) or returns an invalid response, that search uses the original hybrid order
  and docs-mcp-server logs a warning. Search never fails because of the reranker.
- The rerank endpoint only binds to loopback, and the patch only sends chunks to a loopback `http://` URL.
- Result scores become rerank scores (bge: 0..1, jina: -1..1) instead of RRF scores.
- Cost is latency, so the limits are set per device in `[rerank.cpu]` and `[rerank.gpu]` (`candidates`,
  `bge_max_length`, `timeout_ms`); the set for the device the reranker actually runs on is used. bge-reranker-v2-m3
  has 568M parameters: on a CPU 30 full-size chunks at 1024 tokens take close to a minute, hence the CPU default of
  10 candidates at 512 tokens; an RTX 4070 laptop GPU handles 30 at 1024 tokens in about 1.5 s. With `limit = 5`, docs-mcp-server only fetches 10
  full-text candidates; `DOCS_MCP_SEARCH_OVERFETCH_FACTOR` in `[docs_mcp.env]` widens that pool.
- Reranking applies to hybrid search; it is skipped when docs-mcp-server runs without embeddings.

### Quantization

Every model has a `quantization` setting (`[embedding]`, `[rerank]`, `[code_search]`, `[jev]`); the default is
`"int8"`. `"nf4"` (4-bit) uses the least memory, `"none"` loads the model's normal precision. A level the model,
runtime or device cannot do is **not applied**: the model loads normally and the server prints a warning at start-up.

| Model (runtime) | `int8` | `nf4` |
|---|---|---|
| bge-m3, bge-reranker (ONNX) | CPU only (dynamic int8 MatMul/Gather) | CPU and CUDA (`MatMulBnb4`) |
| jina-reranker, bge-code-v1 (PyTorch) | CUDA (bitsandbytes) and CPU (dynamic int8 `Linear`) | CUDA only |
| JEV-9B (PyTorch, CUDA) | yes | yes (its third choice is `"bf16"`) |

- ONNX: onnxruntime runs int8 on NVIDIA GPUs only through TensorRT, so on CUDA/DirectML `int8` keeps the full model
  (the fp16 file for the reranker; for bge-m3 fp16 on CUDA, fp32 on DirectML). `nf4` has no DirectML kernels. A repository's own
  `model_int8.onnx` / `model_quantized.onnx` is used when it has one; otherwise `start.bat setup` builds
  `onnx/model.int8.onnx` or `onnx/model.nf4.onnx` from the fp32 model once (a few minutes, several GB of RAM).
  The vocabulary table stays in full precision under nf4, so XLM-R models shrink less than with int8.
- Measured on bge-m3 against fp32 (cosine similarity of the vectors, CPU): int8 0.98 (2.3 GB -> 540 MB), nf4 0.97
  (-> 1.2 GB).
- bge-m3 on CUDA (`"int8"` or `"none"`): bge-m3 ships no fp16 file, so `start.bat setup` builds
  `onnx/model.fp16.onnx` once from the fp32 model (about 30 s). It fuses attention, SkipLayerNorm and Gelu into
  onnxruntime's CUDA kernels; fused attention keeps memory linear in the text length instead of quadratic. Setup
  prints how many attention layers were fused (24 for bge-m3). Vectors match fp32 (cosine >= 0.9999), so no
  re-scrape is needed. Measured on an RTX 4070 laptop GPU (8 GB): peak VRAM of the embedding server (CUDA context
  included) and time per batch of 2048-token texts:

  | bge-m3 on CUDA | 2 texts per batch (default) | 8 texts per batch |
  |---|---|---|
  | fp32 (before) | 4.6-5.3 GB, 580 ms | 6.8-7.0 GB (the whole card), 3-4.4 s |
  | fp16, fused | 2.1 GB, 100 ms | 2.1 GB, 460 ms |

  The reranker's attention does not fuse (its export uses a pattern onnxruntime does not recognise), so it keeps
  the repository's fp16 file.
- Changing the embedding precision changes the vectors slightly. The launcher remembers the precision the docs index
  was built with and prints a one-time note to re-scrape when it changes; search keeps working meanwhile.
- PyTorch int8 on the CPU uses `torch.ao.quantization`, which PyTorch has deprecated in favour of torchao but still
  ships; if a later PyTorch drops it, the model runs unquantized with a warning.

### jina-reranker-v3.5 (non-commercial only)

jina-reranker-v3.5 is licensed **CC BY-NC 4.0**: personal, non-commercial projects only. It has no ONNX export,
so it runs on PyTorch:

```toml
[rerank]
backend = "jina"
accept_noncommercial_license = true
```

The next start installs `torch` (CPU) and `transformers` (pinned in `pyproject.toml` under
`[project.optional-dependencies] jina`, Windows x64 only, about 1 GB) and downloads the model (~1.2 GB). The model
ships its own Python code (`trust_remote_code`); docs-portable pins it to a reviewed revision (`jina_revision`).
With `[gpu] provider = "cuda"` it runs on the GPU in bfloat16 using PyTorch's CUDA build instead.

## Configuration

Edit `config.toml` (ports, model name, batch size/threads, pinned docs-mcp-server and Node versions, telemetry,
extra docs-mcp-server flags and environment variables such as `GITHUB_TOKEN`). Notes:

- Telemetry of docs-mcp-server is **off** by default here; set `telemetry = true` to enable.
- Changing `model_name`/model makes docs-mcp-server ask to drop existing vectors – vectors from different models are
  incompatible. Re-scrape with `--replace` afterwards.
- **No GPU needed.** Embeddings run on the CPU by default (`onnxruntime`, no PyTorch/CUDA). Expect roughly 3 GB of RAM
  for the model and slower indexing than on a GPU (queries are fast; a large docs site can take a while to index).
  Tune `threads` (CPU threads, `0` = automatic) and, per device, `batch_size`/`max_length` in `[embedding.cpu]` and
  `[embedding.gpu]`. The GPU defaults (2 texts, 2048 tokens) are small on purpose: on Windows a GPU that runs out of
  memory does not fail, it silently spills into system RAM and indexing becomes many times slower.
- **Sharing the CPU.** The model servers never let idle threads spin, so a busy server does not slow down another
  (measured: two onnxruntime servers using all cores ran 2.6x slower than one after the other with spinning on).
  With `threads = 0`, each server uses all physical cores, except when code search also runs on the CPU
  (`[code_search] enabled = true` and `[gpu] provider = "none"`): then bge-code-v1 gets half the cores and the
  embedding server the other half, so code indexing and docs scraping can run together. The reranker keeps all
  cores: its short calls slow down far more on half of them (2.0 s -> 3.3 s per search measured). The launcher
  prints the split; a `threads` value in `[embedding]`, `[rerank]` or `[code_search]` overrides it for that server.

### GPU

One setting in `config.toml` moves the embedding model and the reranker to the GPU:

```toml
[gpu]
provider = "cuda"      # "none" (CPU), "cuda" (NVIDIA) or "directml" (any DirectX 12 GPU)
device_id = 0          # which GPU, if there are several
```

The next `start.bat` swaps in the matching onnxruntime build (`[project.optional-dependencies]` in `pyproject.toml`)
and, for bge, downloads the fp16 reranker instead of int8; on CUDA it also builds a fused fp16 bge-m3 (see [Quantization](#quantization)). The launcher prints where each
model runs (`cpu`, `cuda` or `directml`); if the GPU is not usable it says so and continues on the CPU.

| provider | Package | Extra download | Notes |
|---|---|---|---|
| `cuda` | `onnxruntime-gpu[cuda,cudnn]` | ~2.5 GB (CUDA 13 + cuDNN 9 wheels) | NVIDIA only, recent driver. With `backend = "jina"` also the CUDA build of PyTorch 2.14.1 (~2 GB, pinned by URL + sha256). |
| `directml` | `onnxruntime-directml` | ~25 MB | NVIDIA/AMD/Intel. DirectML is in maintenance mode (last release 1.24.4); jina stays on the CPU. |

To leave VRAM for other apps (browser, IDE), set `[gpu] onnx_memory_limit_mb` (CUDA only, `0` = no cap). Without a
cap, each ONNX model's memory pool grows in power-of-two steps; on an 8 GB card bge-m3 and the reranker each landed
at 1.4 or 2.2 GB from one run to the next. Capping both at 1536 MiB kept all three GPU models (with code search) under
6.7 GB at full load. A request that needs more than the cap fails with an "Available memory" error instead of
spilling into shared memory; raise the cap or lower that model's `batch_size` / `max_length`.

GPU builds are Windows x64 only. Switching back to `"none"` reinstalls the CPU build; the CUDA wheels stay in
`runtime\python` until you delete that folder. Virtual display adapters (e.g. Android emulators) count as GPUs for
DirectML, so set `device_id` if the wrong one is picked.

### JEV-9B judgments (jev-mcp, NVIDIA GPU only)

[jev-mcp](https://github.com/jkudish/jev-mcp) gives an MCP client typed judgment tools (`jev_verify`, `jev_screen`,
`jev_rerank`, `jev_classify`, `jev_decide`, `jev_review`, ...). Instead of TypeSafe's hosted Jev, docs-portable answers
them locally with [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B), an open distillation of Jev 1.13:

```toml
[gpu]
provider = "cuda"

[jev]
enabled = true
quantization = "int8"   # "int8", "nf4" or "bf16"
```

The next start installs PyTorch (CUDA), transformers, accelerate and bitsandbytes (pinned under
`[project.optional-dependencies] jev`), downloads the model pinned to `revision` (~18 GB) and `@jkudish/jev-mcp`
(pinned `mcp_version`, into `runtime\jev-mcp`). At run time the model loads in the background while docs search is
already up; then the launcher prints where its layers went and starts jev-mcp:

```json
{ "mcpServers": { "jev": { "url": "http://127.0.0.1:8090/mcp" } } }
```

- **Memory.** In bf16 the model is 18 GB. Decoder layers go to the GPU while they fit (`gpu_memory_gb`, `0` = free VRAM
  at start-up minus `gpu_reserve_gb`), the rest to RAM (`cpu_memory_gb`) and are streamed to the GPU on every pass;
  anything beyond that is read from disk, which is very slow. `int8`/`nf4` only shrink the layers that stay on the GPU:
  bitsandbytes keeps offloaded layers in bf16. The 2 GB token embedding and the unused `lm_head` stay on disk.
- **Speed.** Several questions share one forward pass (`batch_tokens`), but every pass moves the offloaded layers over
  PCIe, so expect seconds per request rather than Jev's ~0.3 s. `request_timeout_ms` is what jev-mcp waits.
- **Local only.** The System One endpoint (`port`, default 8083) and jev-mcp (`mcp_port`, default 8090) bind to
  loopback. The endpoint requires a bearer token the launcher generates on every start and hands to jev-mcp; jev-mcp
  gets no hosted-provider keys (`TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, ... are removed from its environment).
- **How questions map.** JEV-9B answers yes/no, 0-5 and 2-16 plain options. Noul criteria are appended to the question;
  choice descriptions become `key: description` options, and more than 16 options run as a tournament (groups of up to
  16, then a final); a score with described levels is asked as a choice over the levels and reported as the
  probability-weighted level. States longer than `max_state_tokens` keep their first 60% and last 40%. Images are
  rejected. Answers are close to hosted Jev, not identical (see the model card's limitations).

## Code search (Code page + MCP)

[cocoindex-code](https://github.com/cocoindex-io/cocoindex-code) (Apache-2.0) indexes source code by syntax
(tree-sitter) and searches it by meaning. docs-portable runs it with
[BAAI/bge-code-v1](https://huggingface.co/BAAI/bge-code-v1) (Apache-2.0) and adds a web page to manage it:

```toml
[code_search]
enabled = true
quantization = "int8"   # GPU: "int8" (~2 GB VRAM), "nf4" or "none" (fp16); CPU: "int8" or "none" (float32)
```

The next start installs cocoindex-code and PyTorch (pinned under `[project.optional-dependencies] code` /
`code-embed-cuda`) and downloads the model (~6.2 GB, pinned to `revision`). Then:

- Code page: <http://127.0.0.1:6290>. Add a **local folder** (absolute path) or a **git repository**
  (`https://` URL, optional branch/tag); it is cloned into `runtime\data\code\repos` and indexed. *Re-index* /
  *Pull + index* updates incrementally; *Remove* deletes the index (and the clone for git projects).
- MCP (read-only): `list_projects`, `search_code`

  ```json
  { "mcpServers": { "code-search": { "url": "http://127.0.0.1:6290/mcp" } } }
  ```

How it fits together:

- bge-code-v1 is served on `http://127.0.0.1:8084/v1/embeddings` (`code_embedding_server`). It is asymmetric:
  search queries get the instruction prefix (`query_instruction`), indexed code does not. cocoindex-code calls the
  endpoint through LiteLLM's `nvidia_nim` provider, which passes `input_type` (`query` / `passage`) along.
- The tokenizer comes from the snapshot's `tokenizer.json`; the repository's `tokenization_qwen.py` (remote code) is
  not downloaded or run.
- cocoindex-code's settings, daemon and **indexes** live in `runtime\data\code`; usage tracking is off. A local
  project only gets a small `.cocoindex_code\settings.yml` marker, which is added to the repository's
  `.git\info\exclude` (never to `.gitignore`).
- Git clones are shallow and https-only (`protocol.allow=never` except https, no redirects, no symlinks, no
  submodules). Only hosts in `git_hosts` are allowed; private repositories use git's credential helper (never put
  tokens in the URL).
- Refused local folders: whole drives, your user profile folder itself (subfolders are fine), system folders,
  network/UNC paths and anything that contains `runtime\`.
- The page and the API accept only loopback `Host` headers; changes need the page's per-run CSRF token and a
  same-origin `Origin`. Indexing jobs run one at a time.

Code search is optional like JEV: if it cannot start, docs search keeps running.

## Requirements and limits

- Windows 10 (1803+) or 11, x64 or ARM64, and internet on first run. The batch file only uses `curl.exe` and
  `tar.exe`, which ship with Windows; no PowerShell, no system Python or Node needed.
- The Microsoft Visual C++ runtime (present on virtually every Windows install) is required by `onnxruntime`.
- Move/copy the folder freely, but not while it is running.

## Licenses

This project is MIT licensed. It downloads, at runtime, docs-mcp-server (MIT), BGE-M3 (MIT), bge-reranker-v2-m3
(Apache-2.0), Node.js (MIT), Python (PSF), pip (MIT), onnx (Apache-2.0) and Chromium via Playwright
(BSD-style/Apache-2.0); each keeps
its own license. Only when explicitly enabled: jina-reranker-v3.5 (**CC BY-NC 4.0, non-commercial**), PyTorch
(BSD-3-Clause) and transformers (Apache-2.0); with `[gpu]`: onnxruntime-gpu / onnxruntime-directml (MIT) and the
NVIDIA CUDA/cuDNN runtime wheels (NVIDIA software license); with `[jev]`: autotrust/JEV-9B and its base Qwen3.5-9B
(Apache-2.0), accelerate (Apache-2.0), bitsandbytes (MIT) and jev-mcp (MIT); with `[code_search]`: cocoindex-code
and cocoindex (Apache-2.0), LiteLLM (MIT), BAAI/bge-code-v1 (Apache-2.0) and PyTorch/transformers as above.
