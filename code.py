import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser  # FIXED: Capital P
from langchain_core.runnables.history import RunnableWithMessageHistory  # FIXED: Added import
from langchain_community.chat_message_histories import ChatMessageHistory

import streamlit as st

load_dotenv()


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "messages": [],
        "memory_store": {},
        "model_choice": "groq",
        "session_id": "default"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_memory(session_id: str):
    """
    Get or create memory for a session.
    FIXED: Added () to ChatMessageHistory() - it's a class, needs instantiation!
    """
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = ChatMessageHistory()  # FIXED: Added ()
    return st.session_state.memory_store[session_id]


def get_groq_model(model_name="llama-3.3-70b-versatile"):
    """
    Initialize Groq model.
    FIXED: Added missing api_key definition!
    """
    api_key = os.getenv("GROQ_API_KEY")  # FIXED: This was missing!
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found! Check .env file.")
        return None
    
    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.7,
        max_tokens=2048
    )


def get_openrouter_model(model_name="meta-llama/llama-3.3-70b-instruct:free"):
    """Initialize OpenRouter model"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        st.error("❌ OPENROUTER_API_KEY not found! Check .env file.")
        return None
    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    )


def get_model(provider: str, model_name: str = None):
    """
    Factory function to get the right model.
    FIXED: Typos in model names!
    """
    if provider == "groq":
        # FIXED: "llma" → "llama"
        return get_groq_model(model_name or "llama-3.3-70b-versatile")
    elif provider == "openrouter":
        # FIXED: "instrust" → "instruct"
        return get_openrouter_model(model_name or "meta-llama/llama-3.3-70b-instruct:free")
    else:
        st.error(f"❌ Unknown provider: {provider}")
        return None


def create_chat_chain(llm, session_id: str):
    """
    Create conversation chain with memory.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer concisely but accurately and act as my companion who loves me alot."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    basic_chain = prompt | llm | StrOutputParser()

    # This wrapper automatically handles loading/saving history!
    chain_with_history = RunnableWithMessageHistory(
        basic_chain,
        get_memory,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain_with_history


def main():
    # Initialize
    init_session_state()

    # Page config
    st.set_page_config(
        page_title="Truly Yours",
        page_icon="💕",
        layout="centered"
    )

    st.title("💕 Truly Yours AI")
    st.caption("YOUR FAV AI IS HERE BABES!")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        st.subheader("Choose AI Model")

        provider = st.radio(
            "AI Provider",
            options=["groq", "openrouter"],
            index=0,
            format_func=lambda x: "🚀 Groq (Fast)" if x == "groq" else "🌐 OpenRouter (Many Models)"
        )

        # Model-specific options
        if provider == "groq":
            model = st.selectbox(
                "Model",
                options=[
                    "llama-3.3-70b-versatile",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it"
                ],
                index=0
            )
        else:
            model = st.selectbox(
                "Model",
                options=[
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "google/gemma-2-9b-it:free",
                    "nvidia/llama-3.1-nemotron-70b-instruct:free"
                ],
                index=0
            )

        # Session management
        st.subheader("💾 Conversation")
        session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Change this to start a new conversation. Same ID = same memory."
        )
        st.session_state.session_id = session_id

        # Show memory stats
        memory = get_memory(session_id)
        st.caption(f"Messages in memory: {len(memory.messages)}")

        # Clear memory button
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.memory_store[session_id] = ChatMessageHistory()
            st.session_state.messages = []
            st.rerun()

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):

        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):

                # Initialize AI model
                llm = get_model(provider, model)

                if llm is None:
                    st.error("Failed to initialize AI model. Check your API keys!")
                else:
                    try:
                        # Create the chain with memory
                        chain = create_chat_chain(llm, session_id)

                        # Get response (memory is handled automatically!)
                        response = chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )

                        # Display response
                        st.write(response)

                        # Save to message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Tip: If you hit rate limits, wait a minute or switch models.")


if __name__ == "__main__":
    main()
