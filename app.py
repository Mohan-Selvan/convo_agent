import json
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from backend.agent import get_conversation_history, get_debug_state, run_agent_sync

st.set_page_config(
    page_title="Real-Time Voice RAG Agent",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "sidebar_forced_open" not in st.session_state:
    components.html(
        """
        <script>
        const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute("aria-expanded") === "false") {
            const openBtn = window.parent.document.querySelector('button[aria-label="Open sidebar"]');
            if (openBtn) openBtn.click();
        }
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state.sidebar_forced_open = True
st.title("Real-Time Voice RAG Agent - Phase 3")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_debug" not in st.session_state:
    st.session_state.last_debug = {
        "route": "rag",
        "retrieved_docs": [],
        "tool_outputs": [],
        "latency_ms": None,
    }

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
    with st.chat_message("assistant"):
        st.write_stream(
            run_agent_sync(
                session_id=st.session_state.session_id,
                query=query,
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000
    state = get_debug_state(st.session_state.session_id)
    state["latency_ms"] = latency_ms
    st.session_state.last_debug = state

debug = st.session_state.last_debug
with st.sidebar:
    st.caption(f"Session: {st.session_state.session_id[:8]}")

    with st.expander("Debug", expanded=True):
        st.write("Route")
        st.code(str(debug.get("route", "rag")))

        st.write("Tool outputs")
        tool_outputs = debug.get("tool_outputs", [])
        if tool_outputs:
            st.code(json.dumps(tool_outputs, indent=2, default=str), language="json")
        else:
            st.caption("No tool calls yet.")

        st.write("Retrieved docs")
        docs = debug.get("retrieved_docs", [])
        if docs:
            for doc in docs:
                st.write(f"- {doc}")
        else:
            st.caption("No docs retrieved yet.")

        st.write("Latency")
        latency = debug.get("latency_ms")
        if latency is None:
            st.caption("No request yet.")
        else:
            st.code(f"{latency:.1f} ms")
