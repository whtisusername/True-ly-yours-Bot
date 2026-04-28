import os
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "memory_store": {},
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "session_id": "default"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_memory(session_id: str):
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = ChatMessageHistory()
    return st.session_state.memory_store[session_id]


def get_groq_model(model_name="llama-3.3-70b-versatile"):
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ_API_KEY not found! Check your .env file.")
        logger.warning("GROQ_API_KEY environment variable is missing")
        return None
    
    try:
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.7,
            max_tokens=2048
        )
    except Exception as e:
        logger.error(f"Failed to initialize Groq model: {str(e)}")
        st.error(f"Failed to initialize Groq model: {str(e)}")
        return None


def get_openrouter_model(model_name="meta-llama/llama-3.3-70b-instruct:free"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        st.error("OPENROUTER_API_KEY not found! Check your .env file.")
        logger.warning("OPENROUTER_API_KEY environment variable is missing")
        return None
    
    try:
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter model: {str(e)}")
        st.error(f"Failed to initialize OpenRouter model: {str(e)}")
        return None


def get_model(provider: str, model_name: str = None):
    if provider == "groq":
        return get_groq_model(model_name or "llama-3.3-70b-versatile")
    elif provider == "openrouter":
        return get_openrouter_model(model_name or "meta-llama/llama-3.3-70b-instruct:free")
    else:
        st.error(f"Unknown provider: {provider}")
        logger.error(f"Unknown provider requested: {provider}")
        return None


def create_chat_chain(llm, session_id: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer concisely but accurately and act as a companion who cares about you."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    basic_chain = prompt | llm | StrOutputParser()

    # Wrapper that automatically handles loading/saving history
    chain_with_history = RunnableWithMessageHistory(
        basic_chain,
        get_memory,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain_with_history


def display_chat_history(session_id: str):
    memory = get_memory(session_id)
    
    for message in memory.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)


def validate_input(prompt: str) -> bool:
    if not prompt or not prompt.strip():
        st.warning("⚠️ Please enter a message!")
        return False
    return True


def main():
    """Main Streamlit application"""
    # Initialize
    init_session_state()

    # Page config
    st.set_page_config(
        page_title="Truly Yours",
        page_icon="💕",
        layout="centered"
    )

    st.title("💕 Truly Yours AI")
    st.caption("YOUR FAVORITE AI COMPANION IS HERE! 💜")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        st.subheader("Choose AI Model")

        provider = st.radio(
            "AI Provider",
            options=["groq", "openrouter"],
            index=0 if st.session_state.provider == "groq" else 1,
            format_func=lambda x: "Groq (Fast)" if x == "groq" else "OpenRouter (Many Models)"
        )
        st.session_state.provider = provider

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
        st.session_state.model = model

        # Session management
        st.subheader("Conversation")
        session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Change this to start a new conversation. Same ID = same memory."
        )
        st.session_state.session_id = session_id

        # Show memory stats
        memory = get_memory(session_id)
        message_count = len(memory.messages)
        st.caption(f"Messages in memory: {message_count}")

        # Clear memory button
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.memory_store[session_id] = ChatMessageHistory()
            st.success("Memory cleared!")
            st.rerun()

    # Display chat history from memory (single source of truth)
    display_chat_history(st.session_state.session_id)

    # User input
    if prompt := st.chat_input("Type your message here..."):
        
        # Validate input
        if not validate_input(prompt):
            return

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Initialize AI model
                llm = get_model(st.session_state.provider, st.session_state.model)

                if llm is None:
                    st.error("Failed to initialize AI model. Please check your API keys!")
                else:
                    try:
                        # Create the chain with memory
                        chain = create_chat_chain(llm, st.session_state.session_id)

                        # Get response (memory is handled automatically!)
                        response = chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": st.session_state.session_id}}
                        )

                        # Display response
                        st.write(response)
                        logger.info(f"Response generated successfully for session: {st.session_state.session_id}")

                    except ValueError as e:
                        st.error(f"Input validation error: {str(e)}")
                        logger.error(f"ValueError: {str(e)}")
                    except TimeoutError as e:
                        st.error(f"Request timeout: {str(e)}")
                        st.info("The API took too long to respond. Please try again.")
                        logger.error(f"TimeoutError: {str(e)}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Tip: If you hit rate limits, wait a minute or switch models.")
                        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
