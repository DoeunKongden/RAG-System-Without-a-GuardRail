import time
from uuid import uuid4

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Kongden RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, meta
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "components": {}}


def chat(query: str) -> dict:
    payload = {
        "query": query,
        "conversation_id": st.session_state.conversation_id,
    }
    r = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def run_ingest(topics: list[str], max_articles: int, chunk_strategy: str) -> dict:
    payload = {
        "topics": topics,
        "max_articles": max_articles,
        "chunk_strategy": chunk_strategy,
    }
    r = requests.post(f"{API_BASE}/ingest", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Kongden RAG")
    st.caption("Hybrid search · Rerank · LLaMA-Guard")

    st.divider()

    # Health status
    st.subheader("System Health")
    health = get_health()
    status = health.get("status", "unknown")
    color = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(status, "⚪")
    st.markdown(f"{color} **{status.capitalize()}**")
    for component, ok in health.get("components", {}).items():
        icon = "✅" if ok else "❌"
        st.markdown(f"&nbsp;&nbsp;{icon} {component}")

    st.divider()

    # Conversation controls
    st.subheader("Conversation")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid4())
        st.rerun()
    st.caption(f"ID: `{st.session_state.conversation_id[:8]}…`")

    st.divider()

    # Ingestion panel
    st.subheader("Ingest Documents")
    topics_input = st.text_area(
        "Wikipedia topics (one per line)",
        value="Retrieval-augmented generation\nVector database\nLarge language model",
        height=100,
    )
    max_articles = st.slider("Max articles per topic", 1, 30, 15)
    chunk_strategy = st.selectbox("Chunk strategy", ["recursive", "semantic"])

    if st.button("Run ingestion", use_container_width=True):
        topics = [t.strip() for t in topics_input.splitlines() if t.strip()]
        if not topics:
            st.warning("Enter at least one topic.")
        else:
            with st.spinner("Ingesting…"):
                try:
                    result = run_ingest(topics, max_articles, chunk_strategy)
                    st.success(
                        f"✅ {result['documents_ingested']} docs · "
                        f"{result['chunks_created']} chunks · "
                        f"{result['embeddings_stored']} embeddings "
                        f"({result['duration_ms']:.0f} ms)"
                    )
                    if result.get("errors"):
                        for err in result["errors"]:
                            st.warning(err)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("💬 RAG Chatbot")
st.caption("Grounded answers from your knowledge base · powered by llama3.1:8b + Qdrant")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]

            # Guardrail badge
            if meta.get("guardrail_triggered"):
                st.warning("⚠️ Input guardrail was triggered — this response may be restricted.")

            # Metrics row
            cols = st.columns(3)
            cols[0].metric("Latency", f"{meta['latency_ms']:.0f} ms")
            cols[1].metric("Model", meta.get("model", "—"))
            cols[2].metric("Sources", len(meta.get("sources", [])))

            # Sources expander
            sources = meta.get("sources", [])
            if sources:
                with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        title = src.get("metadata", {}).get("title", "Unknown")
                        url = src.get("metadata", {}).get("source_url", "")
                        score = src.get("relevance_score", 0.0)
                        content = src.get("content", "")

                        st.markdown(
                            f"**{i}. {title}** &nbsp; `{score:.3f}`"
                            + (f" · [{url}]({url})" if url else "")
                        )
                        st.markdown(
                            f"> {content[:400]}{'…' if len(content) > 400 else ''}"
                        )
                        if i < len(sources):
                            st.divider()

# Chat input
if query := st.chat_input("Ask something about your knowledge base…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query, "meta": None})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API and stream response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")

        try:
            response = chat(query)
            answer = response["answer"]
            placeholder.markdown(answer)

            meta = {
                "latency_ms": response.get("latency_ms", 0),
                "model": response.get("model", ""),
                "sources": response.get("sources", []),
                "guardrail_triggered": response.get("guardrail_triggered", False),
            }

            if meta["guardrail_triggered"]:
                st.warning("⚠️ Input guardrail was triggered — this response may be restricted.")

            cols = st.columns(3)
            cols[0].metric("Latency", f"{meta['latency_ms']:.0f} ms")
            cols[1].metric("Model", meta["model"])
            cols[2].metric("Sources", len(meta["sources"]))

            sources = meta["sources"]
            if sources:
                with st.expander(f"📄 {len(sources)} source(s)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        title = src.get("metadata", {}).get("title", "Unknown")
                        url = src.get("metadata", {}).get("source_url", "")
                        score = src.get("relevance_score", 0.0)
                        content = src.get("content", "")

                        st.markdown(
                            f"**{i}. {title}** &nbsp; `{score:.3f}`"
                            + (f" · [{url}]({url})" if url else "")
                        )
                        st.markdown(
                            f"> {content[:400]}{'…' if len(content) > 400 else ''}"
                        )
                        if i < len(sources):
                            st.divider()

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "meta": meta}
            )

        except Exception as e:
            error_msg = f"API error: {e}"
            placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "meta": None}
            )
