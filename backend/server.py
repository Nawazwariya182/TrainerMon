import asyncio
import json
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Header, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from backend.config import API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL_ID, HOST, PORT
from backend.model_engine import engine
from backend.vector_store import vector_store

logger = logging.getLogger("inference_server")
logging.basicConfig(level=logging.INFO)

# Security Setup
security = HTTPBearer(auto_error=False)

def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Enforces API key authentication via Bearer Token or X-API-Key header."""
    token = None
    if auth and auth.credentials:
        token = auth.credentials
    elif x_api_key:
        token = x_api_key

    if not token or token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide Bearer token or 'X-API-Key' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def get_gpu_info() -> Dict[str, Any]:
    """Retrieves live NVIDIA GPU metrics using nvidia-smi with a strict timeout."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2.0)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        if len(parts) >= 6:
            return {
                "available": True,
                "gpu_name": parts[0],
                "memory_total_mb": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_free_mb": float(parts[3]),
                "gpu_utilization_pct": float(parts[4]),
                "temperature_c": float(parts[5]),
            }
    except Exception as e:
        logger.debug(f"Could not read nvidia-smi: {e}")
    return {"available": False, "detail": "NVIDIA GPU metrics unavailable or CPU fallback"}

# Pydantic Schemas
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=8192, description="Max output tokens (8k token cap)")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repeat_penalty: Optional[float] = Field(default=1.12, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CompletionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL_ID
    prompt: str
    max_tokens: Optional[int] = Field(default=8192, description="Max output tokens (8k token cap)")
    temperature: Optional[float] = Field(default=0.7)
    top_p: Optional[float] = Field(default=0.9)
    repeat_penalty: Optional[float] = Field(default=1.12, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class RagIndexRequest(BaseModel):
    project_id: str
    meta: Optional[Dict[str, Any]] = None
    answers: Optional[Dict[str, Any]] = None
    datasets: Optional[List[Dict[str, Any]]] = None
    tech_research: Optional[Dict[str, Any]] = None
    code_files: Optional[Dict[str, str]] = None

class RagQueryRequest(BaseModel):
    project_id: Optional[str] = None
    query: str
    top_k: Optional[int] = 4

class LoadModelRequest(BaseModel):
    model_id: str
    n_ctx: Optional[int] = None
    n_gpu_layers: Optional[int] = 99
    flash_attn: Optional[bool] = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the idle auto-offload background loop (offloads after 2 minutes of inactivity)
    idle_task = asyncio.create_task(engine.idle_reaper_loop())
    logger.info("Startup: Inference engine initialized with 2-minute idle auto-offload policy.")
    yield
    idle_task.cancel()
    logger.info("Shutdown: Unloading models and releasing memory...")
    engine.unload_model()

app = FastAPI(
    title="ML-Train Fast Inference Server",
    description="High-throughput OpenAI-compatible model inference server for local GGUF models with API Key authentication.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check and live hardware telemetry."""
    gpu = get_gpu_info()
    status = engine.get_status()
    return {
        "status": "online",
        "api_key_configured": bool(API_KEY),
        "engine": status,
        "gpu": gpu,
    }

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
def list_models():
    """Returns available models in OpenAI-compatible format."""
    data = []
    for m_id, meta in AVAILABLE_MODELS.items():
        data.append({
            "id": m_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-inference",
            "name": meta["name"],
            "description": meta["description"],
            "default_ctx": meta["default_ctx"],
            "max_ctx": meta["max_ctx"],
            "max_tokens_cap": meta["max_tokens_cap"],
            "is_active": (engine.current_model_id == m_id),
        })
    return {"object": "list", "data": data}

@app.post("/v1/models/load", dependencies=[Depends(verify_api_key)])
def load_model_endpoint(req: LoadModelRequest):
    """Dynamically load or switch an active model."""
    try:
        engine.load_model(
            model_id=req.model_id,
            n_ctx=req.n_ctx,
            n_gpu_layers=req.n_gpu_layers or 99,
            flash_attn=req.flash_attn if req.flash_attn is not None else True,
        )
        return {
            "success": True,
            "message": f"Model '{req.model_id}' successfully loaded into GPU memory.",
            "status": engine.get_status(),
            "gpu": get_gpu_info(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/models/unload", dependencies=[Depends(verify_api_key)])
def unload_model_endpoint():
    """Unload model and free GPU VRAM."""
    engine.unload_model()
    return {"success": True, "message": "Model unloaded. GPU memory freed.", "gpu": get_gpu_info()}

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible Chat Completions endpoint with full streaming support."""
    # Ensure requested model is loaded (or switch from previous model)
    req_model = req.model or DEFAULT_MODEL_ID
    if engine.current_model_id != req_model or engine.llm is None:
        logger.info(
            f"Model request: '{req_model}' (current active: '{engine.current_model_id}', "
            f"loaded: {engine.llm is not None}). Loading model..."
        )
        engine.load_model(req_model)
    else:
        engine.record_activity()

    stop_seqs = [req.stop] if isinstance(req.stop, str) else (req.stop or ["<|im_end|>", "<|endoftext|>"])
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        def stream_generator():
            try:
                response_stream = engine.generate_chat(
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stream=True,
                    stop=stop_seqs,
                )
                for chunk in response_stream:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as err:
                logger.error(f"Streaming error: {err}")
                yield f"data: {json.dumps({'error': str(err)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    t_start = time.time()
    try:
        response = engine.generate_chat(
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=False,
            stop=stop_seqs,
        )
        duration = round(time.time() - t_start, 3)
        if isinstance(response, dict):
            usage = response.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            tok_per_sec = round(completion_tokens / max(duration, 0.001), 2)
            response["inference_metrics"] = {
                "duration_seconds": duration,
                "tokens_per_second": tok_per_sec,
            }
        return response
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
def text_completions(req: CompletionRequest):
    """OpenAI-compatible Text Completions endpoint."""
    req_model = req.model or DEFAULT_MODEL_ID
    if engine.current_model_id != req_model or engine.llm is None:
        logger.info(
            f"Model request: '{req_model}' (current active: '{engine.current_model_id}', "
            f"loaded: {engine.llm is not None}). Loading model..."
        )
        engine.load_model(req_model)
    else:
        engine.record_activity()

    stop_seqs = [req.stop] if isinstance(req.stop, str) else (req.stop or ["<|im_end|>", "<|endoftext|>"])
    try:
        return engine.generate_completion(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stream=req.stream,
            stop=stop_seqs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/research/rag/index", dependencies=[Depends(verify_api_key)])
def index_research_endpoint(req: RagIndexRequest):
    """Indexes project overview, questionnaire answers, datasets, or technical research into vector DB."""
    total_indexed = 0
    if req.meta or req.answers:
        total_indexed += vector_store.index_project_context(req.project_id, req.meta or {}, req.answers or {})
    if req.datasets:
        total_indexed += vector_store.index_datasets(req.project_id, req.datasets)
    if req.tech_research:
        total_indexed += vector_store.index_tech_research(req.project_id, req.tech_research)
    if req.code_files:
        total_indexed += vector_store.index_code_files(req.project_id, req.code_files)

    return {
        "success": True,
        "project_id": req.project_id,
        "indexed_documents": total_indexed,
        "vector_status": vector_store.get_status(),
    }

@app.post("/v1/research/rag/query", dependencies=[Depends(verify_api_key)])
def query_research_endpoint(req: RagQueryRequest):
    """Retrieves top-k context chunks from ChromaDB for RAG-augmented generation."""
    hits = vector_store.query_rag(req.project_id, req.query, top_k=req.top_k or 4)
    return {
        "success": True,
        "query": req.query,
        "hits_count": len(hits),
        "results": hits,
    }

@app.get("/v1/research/rag/status", dependencies=[Depends(verify_api_key)])
def rag_status_endpoint():
    """Returns vector store collection status and storage metrics."""
    return vector_store.get_status()

if __name__ == "__main__":
    import uvicorn
    print(f"=====================================================")
    print(f"🚀 ML-Train Fast Inference Server Starting")
    print(f"🔑 Configured API Key: {API_KEY}")
    print(f"🌐 Listening on: http://{HOST}:{PORT}")
    print(f"📄 Docs available at: http://localhost:{PORT}/docs")
    print(f"=====================================================")
    uvicorn.run("backend.server:app", host=HOST, port=PORT, reload=False)
