# vLLM Production Architecture

## Why the previous setup failed

- **Event loop blocking**: Raw `transformers` + `model.generate()` were run in a thread via `run_in_executor`. With two concurrent requests, two threads called `generate()` on the same in-memory model. PyTorch’s `generate()` is not safe for concurrent use on one model instance, which led to hangs or undefined behavior.
- **No continuous batching**: Each request was a separate inference; the GPU was underused and latency spiked under load.
- **Single global lock**: Serializing with a lock avoided crashes but limited throughput to one request at a time and made the second user wait or hit timeouts.

## How this architecture fixes it

1. **vLLM as inference backend**: vLLM runs the model in a separate process with **continuous batching** and GPU‑friendly scheduling. The FastAPI process never runs GPU code, so the event loop is never blocked by inference.
2. **Async only**: The gateway uses `httpx.AsyncClient` to call the vLLM server. No synchronous `generate()` in the app process; no blocking locks.
3. **Concurrency control**: A semaphore (e.g. 100) limits in‑flight requests. Users beyond that wait in queue; if wait time exceeds `queue_wait_timeout_seconds`, they get **503** with a clear message instead of hanging.
4. **Request queueing**: vLLM’s server queues requests and batches them internally. The gateway adds an application‑level queue (semaphore) so you can support 100+ “concurrent” users while vLLM batches work on the GPU.
5. **Streaming**: SSE streaming is supported; the gateway relays the vLLM server’s event stream to the client with no extra blocking.
6. **Timeouts**: Per‑request and queue‑wait timeouts ensure no request hangs indefinitely.
7. **Graceful shutdown**: Lifespan closes the HTTP client and allows in‑flight requests to finish or time out.

## Port assignment

| Port | Process | Description |
|------|---------|-------------|
| **7000** | **Z-Image main app** (`app.main:app`) | 이미지 생성, 채팅, API — `zimage_webapp/backend/app/` |
| **7001** | vLLM OpenAI server | LLM 추론 전용 (H100) |

**7000번은 메인 앱(`app.main:app`)입니다.** vLLM 전용 게이트웨이(`vllm_server`)가 아님.

## Deployment (two processes)

### 1. vLLM OpenAI server (inference, Linux + H100 NVL) — 포트 **7001**

```bash
# From backend dir (Linux, H100 NVL 1.19.210 등)
cd zimage_webapp/backend
source venv/bin/activate
export VLLM_MODEL=openai/gpt-oss-20b
export VLLM_PORT=7001
# OOM 시: export VLLM_ENFORCE_EAGER=1 또는 VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_NUM_SEQS=32
./scripts/start_vllm_h100.sh
```

**H100 NVL 기본값** (스크립트/환경변수로 조정):

| Parameter | Default (H100 NVL) | Purpose |
|-----------|---------------------|--------|
| `--gpu-memory-utilization` | 0.88 | KV cache and model (OOM 시 0.80) |
| `--max-num-seqs` | 96 | Continuous batching (OOM 시 32) |
| `--max-model-len` | 32768 | Context length |
| `--tensor-parallel-size` | 1 | Single GPU; 2-GPU NVL이면 2 |
| `--dtype` | bfloat16 | H100 NVL 권장 |
| `--trust-remote-code` | true | For custom model code |

Optional: `VLLM_ENFORCE_EAGER=1` to disable CUDA graphs (메모리 절약, 기동 빠름).

### 2. Z-Image main app (FastAPI) — 포트 **7000**

```bash
cd zimage_webapp/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

또는 설정에서 포트 읽기: `PORT=7000` (기본 7000). `run.sh` 사용 시에도 7000번으로 기동됨.

**Optional: vLLM 전용 게이트웨이** (채팅만 별도 서비스로 둘 때): `run_vllm_gateway.py` → 다른 포트(예: 8000)에 두고, 메인 앱(7000)에서 그쪽으로 프록시할 수 있음. 기본 배포는 **7000 = app.main**, **7001 = vLLM**.

## API (OpenAI‑compatible)

- **POST /v1/chat/completions**  
  Body: `{ "model": "gpt-oss-20b", "messages": [...], "stream": true|false, "max_tokens", "temperature" }`  
  - `stream: false` → JSON response.  
  - `stream: true` → SSE (`text/event-stream`).

- **GET /v1/health**  
  Returns `{"status": "ok"}`.

## Performance targets

- **100 concurrent users**: Up to 100 in‑flight requests (semaphore). Additional requests wait; after `queue_wait_timeout_seconds` they receive 503.
- **Stable GPU usage**: vLLM’s continuous batching keeps the H100 utilized.
- **No deadlocks**: No blocking locks in the gateway; only async I/O and semaphore.
- **No hanging**: Queue and request timeouts guarantee a bounded wait or a 503.

## Single‑process (embedded) option

To run the model inside the same process as the gateway (e.g. for simple dev or single‑node deploy), you can use the embedded engine: set `VLLM_USE_EMBEDDED_ENGINE=true` and leave `VLLM_BACKEND_URL` empty. Then the gateway will create an in‑process AsyncLLM engine. This is more sensitive to vLLM version and resource limits; for 100 concurrent users, the two‑process setup above is recommended.
