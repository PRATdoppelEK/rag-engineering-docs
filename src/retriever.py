"""
retriever.py
------------
Semantic retrieval from the FAISS vector store.
Given a natural language query, returns the top-k most relevant document chunks.

Author: Prateek Gaur
"""

import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import load_index, load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


class Retriever:
    """
    Semantic retriever backed by a FAISS index.

    Usage:
        retriever = Retriever()
        results   = retriever.retrieve("What is the recommended mesh density?", top_k=5)
        for r in results:
            print(r["score"], r["source"], r["text"][:200])
    """

    def __init__(self,
                 embed_model: str = None,
                 top_k:       int = None):

        cfg          = load_config()
        self.top_k   = top_k   or cfg["retrieval"]["top_k"]
        model_name   = embed_model or cfg["embedding"]["model"]

        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        logger.info("Loading FAISS index ...")
        self.index, self.chunks = load_index()
        logger.info(f"Retriever ready — {self.index.ntotal} vectors indexed")

    def retrieve(self,
                 query:  str,
                 top_k:  int  = None,
                 min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve the most semantically similar chunks for a given query.

        Args:
            query     : natural language question or keyword string
            top_k     : number of results to return (overrides default)
            min_score : minimum cosine similarity to include a result (0–1)

        Returns:
            List of dicts sorted by score descending:
                {
                    "chunk_id" : int,
                    "source"   : str  (file path),
                    "text"     : str  (chunk content),
                    "score"    : float (cosine similarity, 0–1),
                }
        """
        k = top_k or self.top_k

        # Embed query with same model used for indexing
        q_vec = self.encoder.encode(
            [query],
            normalize_embeddings = True,
            convert_to_numpy     = True,
            show_progress_bar    = False,
        ).astype(np.float32)

        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for empty slots
                continue
            if float(score) < min_score:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source":   chunk["source"],
                "text":     chunk["text"],
                "score":    round(float(score), 4),
            })

        logger.debug(f"Query: '{query[:60]}' → {len(results)} results")
        return results

    def format_context(self,
                       results:        List[Dict],
                       max_total_chars: int = 3000) -> str:
        """
        Concatenate retrieved chunks into a single context string for the LLM prompt.
        Respects a character budget to avoid exceeding context window limits.

        Args:
            results         : list of retrieval results from retrieve()
            max_total_chars : maximum total characters in the context block

        Returns:
            Formatted context string with source citations
        """
        context_parts = []
        total_chars   = 0

        for i, result in enumerate(results, 1):
            source_name = Path(result["source"]).name
            header      = f"[Source {i}: {source_name} | Relevance: {result['score']:.2f}]"
            block       = f"{header}\n{result['text']}"

            if total_chars + len(block) > max_total_chars:
                logger.debug(f"Context budget reached at chunk {i}")
                break

            context_parts.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(context_parts)


# ── CLI — quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the FAISS vector store directly")
    parser.add_argument("--query", type=str,
                        default="What are the battery thermal simulation guidelines?")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = Retriever()
    results   = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.4f} | Source: {Path(r['source']).name}")
        print(f"    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
