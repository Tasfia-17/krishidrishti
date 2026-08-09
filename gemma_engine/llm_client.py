"""
Gemma 4 LLM Client for KrishiDrishti
Communicates with llama-server (llama.cpp) via OpenAI-compatible API.
Supports text and vision (image) input for crop disease analysis.
"""
import base64
import logging
import threading
from typing import Optional, List

import httpx

from config import settings

logger = logging.getLogger("krishidrishti.gemma_engine.llm_client")

# Timeout for inference calls — vision inference can take 12–60s on slow hardware
INFERENCE_TIMEOUT = 300.0
HEALTH_TIMEOUT = 5.0


class InferenceCancelled(Exception):
    """Raised when an in-flight inference is cancelled."""
    pass


# ── Cancellation state ────────────────────────────────────────────────────────
# Allows pre-empting a running inference (e.g., urgent re-analysis request)
_cancel_event = threading.Event()
_active_client: Optional[httpx.Client] = None
_client_lock = threading.Lock()


def cancel_current_inference():
    """
    Cancel any in-flight inference request.
    Closes the active HTTP client so llama-server frees the GPU slot immediately.
    """
    _cancel_event.set()
    with _client_lock:
        if _active_client:
            try:
                _active_client.close()
            except Exception:
                pass
    logger.info("Inference cancelled")


def is_inference_active() -> bool:
    """Check if an inference request is currently in-flight."""
    with _client_lock:
        return _active_client is not None


def _base_url() -> str:
    return settings.llama_server_host.rstrip("/")


def chat(
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Send a chat completion request to llama-server.

    Messages follow OpenAI format:
        [{"role": "user", "content": "text"}]
    or multimodal:
        [{"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ]}]

    Returns the assistant's response text.
    Raises InferenceCancelled if cancel_current_inference() is called mid-request.
    """
    global _active_client

    # Clear stale cancel flag from any previous cancelled request
    _cancel_event.clear()

    url = f"{_base_url()}/v1/chat/completions"
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    client = httpx.Client(timeout=timeout)
    with _client_lock:
        _active_client = client

    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        if _cancel_event.is_set():
            raise InferenceCancelled("Inference cancelled") from e
        raise
    finally:
        with _client_lock:
            if _active_client is client:
                _active_client = None
        try:
            client.close()
        except Exception:
            pass


def chat_with_image(
    prompt: str,
    image_bytes: bytes,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Send a vision chat request with a single image.
    Convenience wrapper used by the crop analyzer.

    Args:
        prompt: Instruction text for Gemma
        image_bytes: JPEG image bytes of the crop photo
        system: Optional system message
        temperature: Sampling temperature (0.0 for deterministic output)
        max_tokens: Maximum tokens in response
    """
    b64 = base64.b64encode(image_bytes).decode()
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    return chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def generate(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: float = INFERENCE_TIMEOUT,
) -> str:
    """
    Simple text generation without vision input.
    Used for advisory text generation and chat Q&A.
    """
    messages = [{"role": "user", "content": prompt}]
    return chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


def is_available() -> bool:
    """Check if llama-server is reachable and healthy."""
    try:
        url = f"{_base_url()}/health"
        response = httpx.get(url, timeout=HEALTH_TIMEOUT)
        return response.status_code == 200
    except Exception:
        return False


def get_server_status() -> dict:
    """Get detailed server status for the health endpoint."""
    try:
        url = f"{_base_url()}/health"
        response = httpx.get(url, timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            return {
                "status": "ok",
                "detail": response.json() if response.text else {},
                "server": _base_url(),
            }
        return {"status": "error", "detail": f"HTTP {response.status_code}"}
    except httpx.ConnectError:
        return {
            "status": "unreachable",
            "detail": f"Cannot connect to llama-server at {_base_url()}",
            "server": _base_url(),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
