# ML-Train Fast Model Inference Server

A high-performance local inference server running quantized GGUF models with FlashAttention and long-context scaling across supported hardware accelerators.

---

## Models Supported
1. **Qwen3-8B-Q8_0** (`qwen3-8b`): High-precision 8B parameter model for coding and general reasoning.
2. **Qwen3.5-9B-Q8_0** (`qwen3.5-9b`): Extended context reasoning model.
3. **Qwen3-VL-8B-Instruct-Q8_0** (`qwen3-vl-8b`): Multimodal Vision-Language model with `mmproj-Qwen3-VL-8B-Instruct-F16.gguf` projector.

---

## Resource Management & Auto-Offload Policy
- **Dynamic Model Switching**: When an API call requests a model different from the currently active model, the server automatically unloads the previous model, clears system memory/caches, and loads the requested model.
- **2-Minute Inactivity Auto-Offload**: If no API requests arrive for more than 2 minutes (120 seconds), the background idle watcher automatically offloads the active model and releases memory. When a new inference request is received, the target model is loaded on-demand.
- **Configurable Idle Timeout**: Adjust `IDLE_TIMEOUT_SECONDS=120` in `backend/.env`.

---

## Authentication
Every request must include your API key configured in `backend/.env`:
```http
Authorization: Bearer <YOUR_API_KEY>
```
Or via header:
```http
X-API-Key: <YOUR_API_KEY>
```

---

## Key Endpoints

### 1. Health & Telemetry
```bash
curl -X GET http://localhost:8000/health
```

### 2. List Models (OpenAI-Compatible)
```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer <YOUR_API_KEY>"
```

### 3. Load / Switch Active Model
```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "qwen3-8b",
    "n_ctx": 16384,
    "n_gpu_layers": 99,
    "flash_attn": true
  }'
```

### 4. Chat Completions (OpenAI-Compatible)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "user", "content": "Explain KV Cache Paging in 2 paragraphs."}
    ],
    "max_tokens": 4096,
    "temperature": 0.7,
    "stream": false
  }'
```

### 5. Automated Benchmark & Test Suite
Runs 4 benchmark calls across all models (quick latency, long-context extraction, high-output token generation, and deep reasoning):
```bash
python -m backend.test_client
```
