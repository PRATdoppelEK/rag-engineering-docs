"""
local_llm.py
------------
Ollama local LLM integration for the RAG pipeline.
Handles model availability checks, prompt formatting, and generation.

Supported models: mistral, llama3, phi3, gemma2
Requires Ollama running locally: https://ollama.ai

Usage:
    llm = OllamaLLM(model="mistral")
    if llm.is_available():
        answer = llm.generate("What is SOH in batteries?")

Author: Prateek Gaur
"""

import logging
import time
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaLLM:
    """
    Wrapper around the Ollama local LLM REST API.

    Why local LLMs?
    Engineering documentation often contains proprietary specifications.
    Routing this through cloud APIs creates data privacy risks.
    Ollama runs fully offline — nothing leaves the machine.

    Args:
        model       : Ollama model name (e.g. 'mistral', 'llama3', 'phi3')
        base_url    : Ollama server URL (default: http://localhost:11434)
        temperature : Sampling temperature — lower = more deterministic (0.0–1.0)
        max_tokens  : Maximum tokens to generate
        timeout     : Request timeout in seconds
    """

    def __init__(self,
                 model:       str   = "mistral",
                 base_url:    str   = OLLAMA_BASE_URL,
                 temperature: float = 0.1,
                 max_tokens:  int   = 1024,
                 timeout:     int   = 120):

        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self._generate_url = f"{self.base_url}/api/generate"
        self._tags_url     = f"{self.base_url}/api/tags"

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable and the model is pulled."""
        try:
            resp = requests.get(self._tags_url, timeout=3)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                logger.warning(
                    f"Ollama is running but model '{self.model}' is not pulled.\n"
                    f"Run: ollama pull {self.model}\n"
                    f"Available models: {models}"
                )
            return available
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama server not reachable at %s\n"
                "Start it with: ollama serve", self.base_url
            )
            return False
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False

    def list_models(self) -> list:
        """Return list of all locally available Ollama models."""
        try:
            resp = requests.get(self._tags_url, timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str,
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from the local LLM.

        Args:
            prompt        : User prompt (can include retrieved context)
            system_prompt : Optional system-level instruction

        Returns:
            Generated text as a string

        Raises:
            RuntimeError  : If Ollama is not running or generation fails
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"

        payload = {
            "model":  self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": ["[USER]", "[SYSTEM]"],
            },
        }

        try:
            t0   = time.time()
            resp = requests.post(
                self._generate_url,
                json    = payload,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            data    = resp.json()
            answer  = data.get("response", "").strip()
            elapsed = time.time() - t0

            logger.info(
                f"Generated {len(answer)} chars in {elapsed:.1f}s "
                f"(model={self.model}, tokens≈{len(answer)//4})"
            )
            return answer

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                f"Start the server: ollama serve\n"
                f"Then pull the model: ollama pull {self.model}"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self):
        status = "available" if self.is_available() else "not available"
        return (
            f"OllamaLLM(model='{self.model}', "
            f"base_url='{self.base_url}', "
            f"status={status})"
        )


# ── CLI quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test local Ollama LLM")
    parser.add_argument("--model",  type=str, default="mistral")
    parser.add_argument("--prompt", type=str,
                        default="In one sentence, what is State of Health in batteries?")
    args = parser.parse_args()

    llm = OllamaLLM(model=args.model)
    print(f"\nOllama status : {llm}")
    print(f"Available models: {llm.list_models()}\n")

    if llm.is_available():
        print(f"Prompt: {args.prompt}\n")
        response = llm.generate(args.prompt)
        print(f"Response:\n{response}")
    else:
        print("Ollama not available — install from https://ollama.ai")
        print("Then run: ollama pull mistral")
