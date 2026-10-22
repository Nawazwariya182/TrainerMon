import asyncio
import gc
import logging
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from backend.config import AVAILABLE_MODELS, DEFAULT_MODEL_ID, IDLE_TIMEOUT_SECONDS

logger = logging.getLogger("model_engine")
logging.basicConfig(level=logging.INFO)

class ModelEngine:
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.current_model_meta: Optional[Dict[str, Any]] = None
        self.llm = None
        self.chat_handler = None
        self.load_time: float = 0.0
        self.last_activity_time: Optional[float] = None
        self.idle_timeout_seconds: int = IDLE_TIMEOUT_SECONDS
        self.lock = threading.Lock()

    def record_activity(self):
        """Refreshes the active timestamp to reset the idle timeout timer."""
        self.last_activity_time = time.time()

    def load_model(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        n_ctx: Optional[int] = None,
        n_gpu_layers: int = 99,
        flash_attn: bool = True,
    ):
        """
        Loads a model into memory.
        If another model is currently loaded, it offloads the previous model first.
        """
        with self.lock:
            if model_id not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model_id: '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}")

            meta = AVAILABLE_MODELS[model_id]
            model_path = meta["path"]
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            # If this exact model is already loaded and active, keep it and reset the timer
            if self.current_model_id == model_id and self.llm is not None:
                logger.info(f"Model '{model_id}' is already loaded and active. Resetting idle timer.")
                self.record_activity()
                return

            # If a different model is currently loaded, offload it before loading the new one
            if self.llm is not None:
                logger.info(
                    f"Model switch detected: Offloading active model '{self.current_model_id}' "
                    f"before loading requested model '{model_id}'..."
                )
                self._do_unload()

            ctx_size = n_ctx or meta["default_ctx"]
            logger.info(f"Loading '{model_id}' from {model_path} (n_ctx={ctx_size}, n_gpu_layers={n_gpu_layers}, flash_attn={flash_attn})...")
            t0 = time.time()

            try:
                from llama_cpp import Llama
            except ImportError:
                raise RuntimeError("llama-cpp-python is not installed or missing local bindings.")

            chat_handler = None
            if meta.get("mmproj") and meta["mmproj"].exists():
                try:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    logger.info(f"Enabling multimodal vision projector: {meta['mmproj']}")
                    chat_handler = Llava15ChatHandler(clip_model_path=str(meta["mmproj"]))
                except Exception as e:
                    logger.warning(f"Could not load vision projector chat handler: {e}")

            # Initialize Llama runtime
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=ctx_size,
                n_gpu_layers=n_gpu_layers,
                flash_attn=flash_attn,
                n_batch=1024,
                n_threads=8,
                chat_handler=chat_handler,
                verbose=False,
            )

            self.current_model_id = model_id
            self.current_model_meta = meta
            self.chat_handler = chat_handler
            self.load_time = round(time.time() - t0, 3)
            self.record_activity()
            logger.info(f"Model '{model_id}' loaded in {self.load_time}s.")

    def _do_unload(self):
        """Internal routine to release model and free memory. Assumes self.lock is acquired."""
        if self.llm is not None:
            offloaded_model = self.current_model_id
            logger.info(f"Offloading model '{offloaded_model}' and freeing memory...")
            try:
                self.llm.close()
            except Exception as e:
                logger.debug(f"Error during llm.close(): {e}")
            self.llm = None
            self.chat_handler = None
            self.current_model_id = None
            self.current_model_meta = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info(f"Model '{offloaded_model}' offloaded successfully.")

    def unload_model(self):
        """Explicitly unloads the active model and frees memory."""
        with self.lock:
            self._do_unload()

    def check_idle_and_offload(self) -> bool:
        """
        Checks if the engine has been idle for longer than the idle timeout (default 120s / 2min).
        If idle, unloads the model to release system resources. Returns True if offloaded.
        """
        with self.lock:
            if self.llm is not None and self.last_activity_time is not None:
                elapsed = time.time() - self.last_activity_time
                if elapsed >= self.idle_timeout_seconds:
                    logger.info(
                        f"Inference idle timeout reached ({int(elapsed)}s >= {self.idle_timeout_seconds}s). "
                        f"Auto-offloading model '{self.current_model_id}' to free resources."
                    )
                    self._do_unload()
                    return True
        return False

    async def idle_reaper_loop(self):
        """Background asynchronous task to continuously inspect and enforce idle offloading."""
        logger.info(f"Idle watcher active: Models will auto-offload if idle for > {self.idle_timeout_seconds} seconds.")
        while True:
            try:
                await asyncio.sleep(5)
                self.check_idle_and_offload()
            except asyncio.CancelledError:
                logger.info("Idle reaper loop cancelled on server shutdown.")
                break
            except Exception as e:
                logger.warning(f"Unexpected error in idle reaper loop: {e}")

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Any:
        """Runs chat completion through the loaded model."""
        with self.lock:
            if self.llm is None:
                raise RuntimeError("No model is currently loaded. Engine must load a model first.")

            self.record_activity()
            cap = self.current_model_meta.get("max_tokens_cap", 8192) if self.current_model_meta else 8192
            target_max_tokens = min(max_tokens or 8192, cap)

            effective_stop = stop or ["<|im_end|>", "<|endoftext|>"]
            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=target_max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.12,
                stream=stream,
                stop=effective_stop,
            )

        if stream:
            def monitored_stream():
                try:
                    for chunk in result:
                        self.record_activity()
                        yield chunk
                finally:
                    self.record_activity()
            return monitored_stream()

        self.record_activity()
        return result

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        stop: Optional[List[str]] = None,
    ) -> Any:
        """Runs raw completion through the loaded model."""
        with self.lock:
            if self.llm is None:
                raise RuntimeError("No model is currently loaded. Engine must load a model first.")

            self.record_activity()
            cap = self.current_model_meta.get("max_tokens_cap", 8192) if self.current_model_meta else 8192
            target_max_tokens = min(max_tokens, cap)

            effective_stop = stop or ["<|im_end|>", "<|endoftext|>"]
            result = self.llm.create_completion(
                prompt=prompt,
                max_tokens=target_max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=1.12,
                stream=stream,
                stop=effective_stop,
            )

        if stream:
            def monitored_stream():
                try:
                    for chunk in result:
                        self.record_activity()
                        yield chunk
                finally:
                    self.record_activity()
            return monitored_stream()

        self.record_activity()
        return result

    def get_status(self) -> Dict[str, Any]:
        """Returns current engine status, model info, and idle metrics."""
        now = time.time()
        idle_seconds = round(now - self.last_activity_time, 1) if self.last_activity_time is not None else None
        return {
            "loaded": self.llm is not None,
            "current_model": self.current_model_id,
            "metadata": self.current_model_meta,
            "load_time_seconds": self.load_time,
            "idle_seconds": idle_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "auto_offloaded": self.llm is None and self.current_model_id is None,
        }

# Global singleton engine
engine = ModelEngine()
