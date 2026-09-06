# PolyRAG — Pinned Versions & Official Docs

Everything pinned/validated against PyPI and official sources. When implementing a module, consult
the linked documentation and keep the pinned versions.

## Pinned versions (Aug 2026)

| Package | Version | Notes |
|---|---|---|
| Python | **3.12.x** | pinned via `.python-version`; use `uv` |
| fastapi | **0.141.1** | not "0.41.1" — that was a typo in planning |
| uvicorn[standard] | **0.52.3** | ASGI server |
| pyyaml | **6.0.3** | config.yaml |
| openai | latest 1.x | async client for all llama-server instances |
| numpy | latest | router/cache math |
| faiss-cpu | **1.15.0** | CAG + router index (CPU, no CUDA) |
| qdrant-client | **1.19.0** | matches Qdrant server v1.19.0 |
| networkx | latest | RAG 3 graph. **API warning:** 3.6+ removed `personalized_pagerank`; `pagerank` needs scipy → use the bundled pure-numpy power-iteration PPR from `rags/graph.py` (see `math.md`) |
| pandas + openpyxl | latest | CSV/XLSX → SQLite |
| opentelemetry-sdk | **1.44.0** | traces |
| opentelemetry-instrumentation-fastapi | resolve with sdk (uv.lock) | auto-instrumentation |
| python-multipart | latest | file uploads |
| httpx | latest | health checks |
| dev: pytest, pytest-asyncio, ruff | latest | |

## Binaries / models

| Item | Version | File | VRAM |
|---|---|---|---|
| llama.cpp (Vulkan) | **b10453** | `llama-b10453-bin-win-vulkan-x64.zip` | — |
| Qdrant standalone | **v1.19.0** | `qdrant-x86_64-pc-windows-msvc.zip` | — |
| Qwen3.5-4B | Q4_K_M | `Qwen3.5-4B-Q4_K_M.gguf` (~9.5GB) | ~9.5GB |
| GLM-OCR | Q8_0 | `GLM-OCR-Q8_0.gguf` (0.89GB) + `mmproj-GLM-OCR-Q8_0.gguf` (0.45GB) | ~2.3GB |
| BGE-M3 | Q8_0 | `bge-m3-Q8_0.gguf` (~0.6GB) | ~0.6GB |

Total resident VRAM ≈ **12.4GB of 16GB** — all 3 models stay loaded. A "load OCR on demand" mode is
an optional optimization for weaker machines, not a requirement.

### Download commands

```powershell
hf download Qwen/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models
hf download ggml-org/GLM-OCR-GGUF GLM-OCR-Q8_0.gguf mmproj-GLM-OCR-Q8_0.gguf --local-dir models
hf download gpustack/bge-m3-GGUF bge-m3-Q8_0.gguf --local-dir models
```

## Official documentation (consult before coding each area)

| Area | Link |
|---|---|
| llama.cpp build/Vulkan | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md |
| llama.cpp multimodal (GLM-OCR, mmproj) | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md |
| llama.cpp server API | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| FastAPI | https://fastapi.tiangolo.com/ |
| OpenAI Python SDK (client) | https://github.com/openai/openai-python |
| PyYAML | https://pyyaml.org/wiki/PyYAMLDocumentation |
| FAISS | https://faiss.ai/ · https://github.com/facebookresearch/faiss/wiki |
| NumPy | https://numpy.org/doc/stable/ |
| Qdrant | https://qdrant.tech/documentation/ |
| SQLite (stdlib) | https://docs.python.org/3/library/sqlite3.html |
| pandas | https://pandas.pydata.org/docs/ |
| networkx / personalized_pagerank | https://networkx.org/documentation/stable/ |
| HippoRAG 2 paper (math + concept) | https://arxiv.org/pdf/2502.14802 |
| HippoRAG repo (reference only) | https://github.com/OSU-NLP-Group/HippoRAG |
| OpenTelemetry Python | https://opentelemetry.io/docs/languages/python/ · https://opentelemetry-python.readthedocs.io/en/latest/ |
| SvelteKit 2 | https://svelte.dev/docs/kit/introduction |
| Tailwind v4 + SvelteKit | https://tailwindcss.com/docs/installation/framework-guides/sveltekit |
| marked | https://www.npmjs.com/package/marked |
| uv | https://docs.astral.sh/uv/ |

## Resource scaling (honest numbers)

- **Storage:** 1M tokens ≈ ~3.2k chunks (512 tok) → ~50–150MB across stores. 5M tokens ≈ ~0.5–1.5GB.
  RAM (32GB) is never the bottleneck; **VRAM is fixed by models, not stored tokens.**
- **Ingestion time is the real cost:** ~5M tokens ≈ 8–24h local (embeddings ~1.5–4h; RAG 3 NER
  generation is the slowest at ~6–18h). A `seed_dataset.py` stress-test generator is planned for
  demo purposes.
