import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from backend.config import API_KEY, AVAILABLE_MODELS, HOST, PORT

SERVER_URL = f"http://127.0.0.1:{PORT}"

class InferenceClient:
    def __init__(self, base_url: str = SERVER_URL, api_key: str = API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def check_health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> List[Dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def load_model(self, model_id: str, n_ctx: Optional[int] = None) -> Dict[str, Any]:
        payload = {"model_id": model_id}
        if n_ctx:
            payload["n_ctx"] = n_ctx
        resp = requests.post(f"{self.base_url}/v1/models/load", json=payload, headers=self.headers, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=600,
        )
        total_time = round(time.time() - t0, 3)
        resp.raise_for_status()
        data = resp.json()
        data["client_duration_sec"] = total_time
        return data

def run_tests_for_model(client: InferenceClient, model_id: str):
    print(f"\n======================================================================")
    print(f"🔬 RUNNING BENCHMARK & INFERENCE TEST SUITE FOR: {model_id.upper()}")
    print(f"======================================================================")

    # Step 1: Switch / Load model
    print(f"⚙️ [Step 0] Loading model '{model_id}' into GPU memory...")
    try:
        load_res = client.load_model(model_id)
        print(f"  ✓ Model Loaded: {load_res.get('message')}")
        gpu_info = load_res.get("gpu", {})
        if gpu_info.get("available"):
            print(f"  ⚡ GPU VRAM Used: {gpu_info.get('memory_used_mb')} MB / {gpu_info.get('memory_total_mb')} MB")
    except Exception as e:
        print(f"  ❌ Failed to load model '{model_id}': {e}")
        return

    # Call 1: Fast Latency & Throughput Benchmark
    print(f"\n▶ [Test 1/4] Quick Latency & Speed Benchmark")
    t1_messages = [
        {"role": "system", "content": "You are a concise, ultra-fast AI assistant."},
        {"role": "user", "content": "Explain in 3 bullet points why FlashAttention is critical for high-speed LLM inference."},
    ]
    try:
        res1 = client.chat_completion(model=model_id, messages=t1_messages, max_tokens=512, temperature=0.3)
        metrics = res1.get("inference_metrics", {})
        usage = res1.get("usage", {})
        content = res1["choices"][0]["message"]["content"]
        
        print(f"  ✓ Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  ✓ Duration: {metrics.get('duration_seconds', res1.get('client_duration_sec'))}s")
        print(f"  ⚡ Throughput: {metrics.get('tokens_per_second', 'N/A')} tokens/sec")
        print(f"  💬 Output snippet:\n    " + content.replace("\n", "\n    ")[:300] + ("..." if len(content) > 300 else ""))
    except Exception as e:
        print(f"  ❌ Test 1 failed: {e}")

    # Call 2: Long-Context Processing Test
    print(f"\n▶ [Test 2/4] Long Context Ingestion & Information Extraction")
    # Generate ~2,000 words of background context with a hidden critical directive
    context_paragraphs = [
        f"Section {i}: In high-throughput distributed machine learning training architectures, "
        f"gradient synchronizations must be carefully interleaved with pipeline bubbles. "
        f"Worker node cluster {i} reports memory pressure coefficients of {0.05 * i:.2f}."
        for i in range(1, 80)
    ]
    needle = "SECRET_DIRECTIVE_ALPHA: All worker nodes must execute checkpoint commit at exactly epoch 42."
    context_paragraphs.insert(40, needle)
    long_document = "\n".join(context_paragraphs)

    t2_messages = [
        {"role": "system", "content": "You are an analytical assistant capable of ingesting long complex contexts."},
        {"role": "user", "content": f"Below is a long cluster telemetry report:\n\n{long_document}\n\nQuestion: What is the exact SECRET_DIRECTIVE_ALPHA mentioned in the text?"},
    ]
    try:
        res2 = client.chat_completion(model=model_id, messages=t2_messages, max_tokens=512, temperature=0.1)
        usage = res2.get("usage", {})
        content = res2["choices"][0]["message"]["content"]
        print(f"  ✓ Prompt Tokens Ingested: {usage.get('prompt_tokens', 'N/A')}")
        print(f"  ✓ Duration: {res2.get('client_duration_sec')}s")
        print(f"  💬 Output:\n    " + content.replace("\n", "\n    ").strip())
    except Exception as e:
        print(f"  ❌ Test 2 failed: {e}")

    # Call 3: High Generation Output Cap Test
    print(f"\n▶ [Test 3/4] Max Generation Cap Test (High Output Token Generation)")
    t3_messages = [
        {"role": "system", "content": "You are a comprehensive technical author. Write detailed, extensive, in-depth documentation without stopping early."},
        {"role": "user", "content": "Provide a complete, production-grade guide to building a scalable LLM inference pipeline. Detail: 1) KV Cache Paging, 2) Continuous Batching, 3) Speculative Decoding, 4) Quantization (FP8, AWQ, GGUF), and 5) Production Monitoring. Write as much thorough detail as possible."},
    ]
    try:
        res3 = client.chat_completion(model=model_id, messages=t3_messages, max_tokens=4096, temperature=0.7)
        usage = res3.get("usage", {})
        metrics = res3.get("inference_metrics", {})
        content = res3["choices"][0]["message"]["content"]
        print(f"  ✓ Completion Tokens Generated: {usage.get('completion_tokens', 'N/A')} tokens")
        print(f"  ⚡ Generation Speed: {metrics.get('tokens_per_second', 'N/A')} tokens/sec")
        print(f"  ⏱️ Total Generation Time: {res3.get('client_duration_sec')}s")
        print(f"  💬 Snippet (first 250 chars):\n    " + content[:250].replace("\n", " ") + "...")
    except Exception as e:
        print(f"  ❌ Test 3 failed: {e}")

    # Call 4: Deep Reasoning / Specialized Modality Test
    print(f"\n▶ [Test 4/4] Advanced Algorithmic Reasoning & Multi-turn Capability")
    t4_messages = [
        {"role": "system", "content": "You are an elite competitive programmer and AI researcher."},
        {"role": "user", "content": "Write a complete Python implementation of an LRU Cache with O(1) get and put, thread-safe using locks, with TTL (time to live) expiration per key and automatic background eviction. Include type hints and clean docstrings."},
    ]
    try:
        res4 = client.chat_completion(model=model_id, messages=t4_messages, max_tokens=2048, temperature=0.2)
        usage = res4.get("usage", {})
        content = res4["choices"][0]["message"]["content"]
        print(f"  ✓ Completion Tokens: {usage.get('completion_tokens', 'N/A')}")
        print(f"  ✓ Duration: {res4.get('client_duration_sec')}s")
        print(f"  💬 Generated Implementation Snippet:\n" + "\n".join(["    " + line for line in content.splitlines()[:15]]) + "\n    ...")
    except Exception as e:
        print(f"  ❌ Test 4 failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test client for ML-Train Inference Server")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (qwen3-8b, qwen3.5-9b, qwen3-vl-8b)")
    parser.add_argument("--url", type=str, default=SERVER_URL, help="Server base URL")
    args = parser.parse_args()

    client = InferenceClient(base_url=args.url)
    
    print(f"📡 Connecting to inference server at: {client.base_url}")
    print(f"🔑 Using API Key: {client.api_key[:8]}...{client.api_key[-4:]}")
    
    try:
        health = client.check_health()
        print(f"✅ Server Health: ONLINE")
        print(f"   Loaded Model: {health.get('engine', {}).get('current_model')}")
        gpu = health.get("gpu", {})
        if gpu.get("available"):
            print(f"   GPU: {gpu.get('gpu_name')} | VRAM: {gpu.get('memory_used_mb')} / {gpu.get('memory_total_mb')} MB | Util: {gpu.get('gpu_utilization_pct')}%")
    except Exception as e:
        print(f"❌ Could not connect to inference server at {client.base_url}: {e}")
        print("Please start the server first with: python -m backend.server")
        sys.exit(1)

    models_to_test = [args.model] if args.model else list(AVAILABLE_MODELS.keys())
    for mid in models_to_test:
        run_tests_for_model(client, mid)

    print(f"\n======================================================================")
    print(f"🎉 ALL INFERENCE BENCHMARK TESTS COMPLETED SUCCESSFULLY!")
    print(f"======================================================================\n")

if __name__ == "__main__":
    main()
