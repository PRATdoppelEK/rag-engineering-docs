"""
ingest.py
---------
Document ingestion pipeline: load → chunk → embed → store in FAISS index.

Supported formats: PDF, DOCX, TXT, Markdown
Run this once to build the vector store before querying.

Usage:
    python src/ingest.py --docs_path data/sample_docs/
    python src/ingest.py --docs_path /path/to/your/pdfs/ --chunk_size 512

Author: Prateek Gaur
"""

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CONFIG     = ROOT / "config.yaml"
INDEX_DIR  = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH    = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG.exists():
        with open(CONFIG) as f:
            return yaml.safe_load(f)
    # Defaults if config.yaml missing
    return {
        "chunking":   {"chunk_size": 512, "chunk_overlap": 64},
        "embedding":  {"model": "all-MiniLM-L6-v2"},
        "retrieval":  {"top_k": 5},
    }


# ── Document loading ───────────────────────────────────────────────────────────

def load_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
        return ""


def load_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc  = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except ImportError:
        logger.warning("python-docx not installed. Install with: pip install python-docx")
        return ""


def load_document(path: Path) -> str:
    """Route to the correct loader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in (".txt", ".md", ".rst"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        logger.warning(f"Unsupported file type: {ext} — skipping {path.name}")
        return ""


def load_all_documents(docs_path: Path) -> List[Dict]:
    """
    Walk docs_path recursively and load all supported documents.

    Returns:
        List of dicts: {"source": str, "text": str}
    """
    supported = {".pdf", ".docx", ".txt", ".md", ".rst"}
    docs      = []

    for file_path in sorted(docs_path.rglob("*")):
        if file_path.suffix.lower() not in supported:
            continue
        text = load_document(file_path)
        if len(text.strip()) < 50:
            logger.warning(f"Skipped (too short): {file_path.name}")
            continue
        docs.append({"source": str(file_path), "text": text})
        logger.info(f"  Loaded: {file_path.name} ({len(text):,} chars)")

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str,
               chunk_size:    int = 512,
               chunk_overlap: int = 64) -> List[str]:
    """
    Split text into overlapping windows of approximately `chunk_size` characters.
    Tries to split on sentence boundaries (". ") to preserve semantic coherence.

    Args:
        text          : raw document text
        chunk_size    : target chunk size in characters
        chunk_overlap : overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks    = []
    current   = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = current + ". " + sentence if current else sentence

        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            # Keep last `chunk_overlap` chars as context carry-over
            current = current[-chunk_overlap:] + " " + sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    # Filter out very short chunks
    return [c for c in chunks if len(c) > 30]


def chunk_documents(docs: List[Dict],
                    chunk_size:    int = 512,
                    chunk_overlap: int = 64) -> List[Dict]:
    """
    Chunk all loaded documents.

    Returns:
        List of dicts: {"chunk_id": int, "source": str, "text": str}
    """
    all_chunks = []
    chunk_id   = 0

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for c in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source":   doc["source"],
                "text":     c,
            })
            chunk_id += 1

    logger.info(f"Total chunks created: {len(all_chunks)} "
                f"(avg {len(all_chunks)/max(len(docs),1):.0f} per document)")
    return all_chunks


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_chunks(chunks: List[Dict],
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 64) -> np.ndarray:
    """
    Compute sentence embeddings for all chunks.

    Args:
        chunks     : list of chunk dicts (must have "text" key)
        model_name : SentenceTransformer model identifier
        batch_size : encoding batch size

    Returns:
        embeddings : np.ndarray of shape (n_chunks, embedding_dim), float32
    """
    logger.info(f"Loading embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)

    texts  = [c["text"] for c in chunks]
    t0     = time.time()

    logger.info(f"Embedding {len(texts)} chunks ...")
    embeddings = encoder.encode(
        texts,
        batch_size        = batch_size,
        show_progress_bar = True,
        normalize_embeddings = True,   # cosine similarity via dot product
        convert_to_numpy  = True,
    )
    elapsed = time.time() - t0
    logger.info(
        f"Embedding complete: {embeddings.shape} in {elapsed:.1f}s "
        f"({len(texts)/elapsed:.0f} chunks/sec)"
    )
    return embeddings.astype(np.float32)


# ── FAISS index ────────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product = cosine similarity for
    unit-normalised vectors). Best for exact search on small-to-medium corpora.

    For large corpora (>100k chunks), switch to IndexIVFFlat with nlist=128.

    Args:
        embeddings : (n, dim) float32 array, L2-normalised

    Returns:
        FAISS index with all vectors added
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.Index, chunks: List[Dict]):
    """Persist the FAISS index and chunk metadata to disk."""
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Index saved to {INDEX_PATH}")
    logger.info(f"Metadata saved to {METADATA_PATH}")


def load_index():
    """Load a previously built FAISS index and metadata from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run ingest.py first."
        )
    index  = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Index loaded: {index.ntotal} vectors")
    return index, chunks


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_ingestion(docs_path: Path,
                  chunk_size:    int = 512,
                  chunk_overlap: int = 64,
                  embed_model:   str = "all-MiniLM-L6-v2"):
    """
    Full ingestion pipeline:
        Load documents → Chunk → Embed → Build FAISS index → Save
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"Ingestion pipeline starting")
    logger.info(f"  Source: {docs_path}")
    logger.info(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    logger.info(f"  Embedding model: {embed_model}")
    logger.info(f"{'='*55}\n")

    t0   = time.time()
    docs = load_all_documents(docs_path)

    if not docs:
        logger.error("No documents loaded. Check that docs_path contains supported files.")
        return

    chunks     = chunk_documents(docs, chunk_size, chunk_overlap)
    embeddings = embed_chunks(chunks, embed_model)
    index      = build_faiss_index(embeddings)
    save_index(index, chunks)

    elapsed = time.time() - t0
    logger.info(f"\nIngestion complete in {elapsed:.1f}s")
    logger.info(f"Ready to query: {len(chunks)} chunks indexed")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Ingest documents into FAISS vector store")
    parser.add_argument("--docs_path",     type=Path,
                        default=ROOT / "data" / "sample_docs",
                        help="Directory containing documents to ingest")
    parser.add_argument("--chunk_size",    type=int,
                        default=cfg["chunking"]["chunk_size"])
    parser.add_argument("--chunk_overlap", type=int,
                        default=cfg["chunking"]["chunk_overlap"])
    parser.add_argument("--embed_model",   type=str,
                        default=cfg["embedding"]["model"])
    args = parser.parse_args()

    run_ingestion(
        docs_path     = args.docs_path,
        chunk_size    = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        embed_model   = args.embed_model,
    )


if __name__ == "__main__":
    main()
