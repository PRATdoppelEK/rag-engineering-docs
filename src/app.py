"""
app.py
------
Optional Gradio web interface for the RAG pipeline.
Provides a browser-based chat UI for querying engineering documents.

Usage:
    python src/app.py
    # Opens at http://localhost:7860

Requirements:
    pip install gradio
    Run ingest.py first to build the vector index.

Author: Prateek Gaur
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio")
    sys.exit(1)

from rag_chain import RAGChain
from ingest import load_config

# ── Load pipeline once at startup ─────────────────────────────────────────────

cfg = load_config()

try:
    chain = RAGChain(
        llm_model   = cfg.get("llm", {}).get("model", "mistral"),
        top_k       = cfg["retrieval"]["top_k"],
    )
    llm_available = chain.llm.is_available()
    logger.info(f"RAG chain loaded. LLM available: {llm_available}")
except FileNotFoundError:
    logger.error(
        "No FAISS index found. Run ingest.py first:\n"
        "  python src/ingest.py --docs_path data/sample_docs/"
    )
    chain = None
    llm_available = False


# ── Chat function ──────────────────────────────────────────────────────────────

def chat(message: str,
         history: list,
         top_k: int,
         retrieval_only: bool) -> tuple:
    """
    Process a user query through the RAG pipeline.

    Args:
        message        : user's question
        history        : Gradio chat history (list of [user, assistant] pairs)
        top_k          : number of chunks to retrieve
        retrieval_only : skip LLM, show only retrieved context

    Returns:
        ("", updated_history) — clears input box and updates chat
    """
    if not message.strip():
        return "", history

    if chain is None:
        response = (
            "⚠️ No document index found.\n"
            "Run first: `python src/ingest.py --docs_path data/sample_docs/`"
        )
        history.append([message, response])
        return "", history

    try:
        result = chain.query(
            question       = message,
            top_k          = top_k,
            retrieval_only = retrieval_only or not llm_available,
        )

        if retrieval_only or not llm_available:
            # Format retrieved chunks clearly
            lines = []
            if not llm_available:
                lines.append(
                    "ℹ️ *LLM not available — showing retrieved context only.*\n"
                    "Install Ollama and run `ollama pull mistral` to enable generation.\n"
                )
            for i, chunk in enumerate(result["chunks"], 1):
                src   = Path(chunk["source"]).name
                score = chunk["score"]
                text  = chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else "")
                lines.append(f"**[{i}] {src}** (relevance: {score:.3f})\n{text}")
            response = "\n\n---\n\n".join(lines)
        else:
            response = result["answer"]
            if result["sources"]:
                source_list = "\n".join(
                    f"• {s['source']} (relevance: {s['score']:.3f})"
                    for s in result["sources"]
                )
                response += f"\n\n**Sources:**\n{source_list}"

    except Exception as e:
        response = f"Error: {e}"
        logger.error(f"Query failed: {e}")

    history.append([message, response])
    return "", history


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="RAG — Engineering Document Assistant",
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# Engineering Document Assistant\n"
            "Ask questions about your engineering documentation. "
            "Answers are grounded in the indexed documents — no hallucination."
        )

        # Status banner
        status_color = "green" if llm_available else "orange"
        status_text  = (
            f"✅ LLM active: `{cfg.get('llm', {}).get('model', 'mistral')}`"
            if llm_available
            else "⚠️ LLM not available — retrieval-only mode. Install Ollama to enable generation."
        )
        gr.Markdown(f"<span style='color:{status_color}'>{status_text}</span>")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label  = "Conversation",
                    height = 500,
                    bubble_full_width = False,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder = "Ask a question about your engineering documents...",
                        label       = "",
                        scale       = 8,
                        autofocus   = True,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                top_k_slider = gr.Slider(
                    minimum = 1,
                    maximum = 10,
                    value   = cfg["retrieval"]["top_k"],
                    step    = 1,
                    label   = "Chunks to retrieve (top-k)",
                )
                retrieval_only_toggle = gr.Checkbox(
                    value = not llm_available,
                    label = "Retrieval only (skip LLM)",
                )
                gr.Markdown("### Example queries")
                examples = gr.Examples(
                    examples = [
                        ["What is State of Health in batteries?"],
                        ["What are the thermal simulation guidelines?"],
                        ["Explain the SOC estimation method"],
                        ["What mesh density is recommended for HV battery simulation?"],
                        ["Describe the cell balancing strategy"],
                    ],
                    inputs  = msg_box,
                    label   = "",
                )
                clear_btn = gr.Button("Clear conversation", variant="secondary")

        # ── Event handlers ────────────────────────────────────────────────────
        submit_args = dict(
            fn      = chat,
            inputs  = [msg_box, chatbot, top_k_slider, retrieval_only_toggle],
            outputs = [msg_box, chatbot],
        )
        msg_box.submit(**submit_args)
        send_btn.click(**submit_args)
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch RAG Gradio app")
    parser.add_argument("--host",  type=str, default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    app = build_ui()
    logger.info(f"Launching app at http://{args.host}:{args.port}")
    app.launch(server_name=args.host,
               server_port=args.port,
               share=args.share)
