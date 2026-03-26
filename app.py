import streamlit as st

from backend.agent import run_agent_sync

st.set_page_config(page_title="Real-Time Voice RAG Agent", page_icon="AI")
st.title("Real-Time Voice RAG Agent - Phase 1")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Toggles")
    use_rag = st.toggle("RAG", value=False)
    use_tools = st.toggle("Tools", value=False)

st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response = st.write_stream(
            run_agent_sync(
                query=query,
                history=st.session_state.messages,
                use_rag=use_rag,
                use_tools=use_tools,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
