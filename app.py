import time
import uuid

import streamlit as st

from backend.agent import build_context, get_conversation_history, run_agent_sync

st.set_page_config(page_title="Real-Time Voice RAG Agent", page_icon="AI")
st.title("Real-Time Voice RAG Agent - Phase 2")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_debug" not in st.session_state:
    st.session_state.last_debug = {
        "retrieved_docs": [],
        "tool_outputs": [],
        "latency_ms": None,
    }

with st.sidebar:
    st.subheader("Toggles")
    use_rag = st.toggle("RAG", value=True)
    use_tools = st.toggle("Tools", value=False)
    st.caption(f"Session: {st.session_state.session_id[:8]}")

st.subheader("Chat")

messages = get_conversation_history(st.session_state.session_id)
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    started = time.perf_counter()
    context_payload = build_context(
        query=query,
        use_rag=use_rag,
        use_tools=use_tools,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(
            run_agent_sync(
                session_id=st.session_state.session_id,
                query=query,
                use_rag=use_rag,
                use_tools=use_tools,
                context_payload=context_payload,
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000
    st.session_state.last_debug = {
        "retrieved_docs": context_payload.get("retrieved_docs", []),
        "tool_outputs": context_payload.get("tool_outputs", []),
        "latency_ms": latency_ms,
    }

st.subheader("Debug")
debug = st.session_state.last_debug

with st.expander("Retrieved docs", expanded=True):
    docs = debug.get("retrieved_docs", [])
    if docs:
        for doc in docs:
            st.write(f"- {doc}")
    else:
        st.caption("No docs retrieved yet.")

with st.expander("Latency", expanded=False):
    latency = debug.get("latency_ms")
    if latency is None:
        st.caption("No request yet.")
    else:
        st.write(f"{latency:.1f} ms")
