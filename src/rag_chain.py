"""
rag_chain.py
------------
Full RAG pipeline: query → retrieve → build prompt → generate answer via local LLM.

Supports two LLM backends:
    1. Ollama   — local models (Mistral, LLaMA 3, Phi-3) via REST API
    2. Fallback — retrieval-only mode (returns context without generation)
                  useful when no local LLM is installed

Usage:
    python src/rag_chain.py --query "What mesh density for HV battery simulation?"
    python src/rag_chain.py --query "Explain SOH calculation" --top_k 3
    python src/rag_chain.py --retrieval_only   # skip LLM, just show retrieved chunks

Author: Prateek Gaur
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests

from retriever import Retriever
from ingest import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Prompt template ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant specialised in engineering documentation,
battery systems, and simulation methodologies.

Your task is to answer the user's question using ONLY the information provided in the
context below. Do not use any external knowledge beyond what is given.

Rules:
- Be concise and technical.
- If the context does not contain enough information to answer, say:
  "The provided documents do not contain sufficient information to answer this question."
- Always cite which source (e.g. [Source 1], [Source 2]) your answer comes from.
- Do not hallucinate facts, numbers, or procedures not present in the context.
"""

USER_PROMPT_TEMPLATE = """Context from engineering documents:
{context}

---

Question: {question}

Answer (cite sources):"""


# ── Ollama LLM backend ─────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Lightweight wrapper around the Ollama local LLM API.
    Ollama must be running: https://ollama.ai

    Tested models: mistral, llama3, phi3, gemma2
    """

    def __init__(self,
                 model:       str   = "mistral",
                 base_url:    str   = "http://localhost:11434",
                 temperature: float = 0.1,
                 max_tokens:  int   = 1024):
        self.model       = model
        self.base_url    = base_url
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.endpoint    = f"{base_url}/api/generate"

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the local Ollama model and return the generated text.

        Args:
            prompt : full prompt string (system + context + question)

        Returns:
            Generated answer as a string
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            t0   = time.time()
            resp = requests.post(self.endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            data    = resp.json()
            elapsed = time.time() - t0
            answer  = data.get("response", "").strip()
            logger.info(f"LLM generated {len(answer)} chars in {elapsed:.1f}s")
            return answer
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running? "
                "Start with: ollama serve\n"
                "Then pull a model: ollama pull mistral"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")


# ── RAG chain ──────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Full Retrieval-Augmented Generation pipeline.

    Flow:
        1. Embed query
        2. Retrieve top-k relevant chunks from FAISS
        3. Build prompt: system + retrieved context + user question
        4. Generate answer via local LLM
        5. Return answer + source citations

    Usage:
        chain = RAGChain()
        result = chain.query("What is the recommended SOH threshold for EV batteries?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(self,
                 llm_model:   str = None,
                 top_k:       int = None,
                 temperature: float = None):

        cfg           = load_config()
        top_k         = top_k or cfg["retrieval"]["top_k"]
        llm_model     = llm_model or cfg["llm"]["model"]
        temperature   = temperature or cfg.get("llm", {}).get("temperature", 0.1)

        logger.info("Initialising RAG chain ...")
        self.retriever = Retriever(top_k=top_k)
        self.llm       = OllamaLLM(model=llm_model, temperature=temperature)

        if self.llm.is_available():
            logger.info(f"LLM backend: Ollama ({llm_model}) ✓")
        else:
            logger.warning(
                "Ollama is not running. Retrieval-only mode active.\n"
                "To enable generation: install Ollama and run 'ollama pull mistral'"
            )

    def query(self,
              question:        str,
              top_k:           Optional[int] = None,
              retrieval_only:  bool = False) -> Dict:
        """
        Run the full RAG pipeline for a given question.

        Args:
            question       : user's natural language question
            top_k          : number of chunks to retrieve (overrides default)
            retrieval_only : if True, skip LLM and return only retrieved context

        Returns:
            dict with keys:
                "question"  : str
                "answer"    : str (or retrieved context if retrieval_only)
                "sources"   : list of {"source": str, "score": float}
                "chunks"    : list of raw retrieved chunks
        """
        logger.info(f"\nQuery: {question}")

        # Step 1: Retrieve
        results = self.retriever.retrieve(question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer":   "No relevant documents found in the knowledge base.",
                "sources":  [],
                "chunks":   [],
            }

        # Step 2: Format context
        context = self.retriever.format_context(results)
        sources = [
            {"source": Path(r["source"]).name, "score": r["score"]}
            for r in results
        ]

        # Step 3: Retrieval-only mode
        if retrieval_only or not self.llm.is_available():
            return {
                "question": question,
                "answer":   context,
                "sources":  sources,
                "chunks":   results,
            }

        # Step 4: Build prompt and generate
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            + USER_PROMPT_TEMPLATE.format(context=context, question=question)
        )
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "chunks":   results,
        }

    def format_response(self, result: Dict) -> str:
        """Pretty-print a query result for terminal output."""
        lines = [
            f"\n{'='*60}",
            f"Question: {result['question']}",
            f"{'='*60}",
            f"\nAnswer:\n{result['answer']}",
            f"\n{'─'*60}",
            f"Sources used:",
        ]
        for i, src in enumerate(result["sources"], 1):
            lines.append(f"  [{i}] {src['source']}  (relevance: {src['score']:.3f})")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("--query",          type=str,
                        default="What are the thermal simulation guidelines for battery packs?")
    parser.add_argument("--top_k",          type=int,
                        default=cfg["retrieval"]["top_k"])
    parser.add_argument("--llm_model",      type=str,
                        default=cfg.get("llm", {}).get("model", "mistral"))
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Skip LLM generation — show retrieved chunks only")
    parser.add_argument("--temperature",    type=float, default=0.1)
    args = parser.parse_args()

    chain  = RAGChain(llm_model   = args.llm_model,
                      top_k       = args.top_k,
                      temperature = args.temperature)

    result = chain.query(args.query, retrieval_only=args.retrieval_only)
    print(chain.format_response(result))


if __name__ == "__main__":
    main()
