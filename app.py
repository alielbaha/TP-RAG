
import streamlit as st
from pathlib import Path

from src.retriever import Retriever
from src.llm_handler import LLMQuestionAnswering
from src.utils import load_config

st.set_page_config(
    page_title="RAG Chat",
    page_icon="",
    layout="centered"
)

st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_system():
  
    try:
        # Load config
        config = load_config("config.yaml")
        
        # Initialize retriever
        retriever = Retriever(
            vector_store_path=config['indexing']['vector_store']['path'],
            embedding_model_name=config['indexing']['embedding_model'],
            collection_name=config['indexing']['vector_store']['collection_name'],
            top_k=config['retrieval']['top_k']
        )
        retriever.load_vector_store()
        
        # Initialize LLM
        llm_handler = LLMQuestionAnswering(
            model_name=config['llm']['model_name'],
            device=config['llm']['device'],
            max_new_tokens=config['llm']['max_new_tokens'],
            temperature=config['llm']['temperature'],
            use_api=config['llm'].get('use_api', False),
            api_token=config['llm'].get('api_token')
        )
        llm_handler.load_model()
        
        # Create prompt template
        prompt_template = llm_handler.create_prompt_template(
            template=config['prompt']['template']
        )
        
        return retriever, llm_handler, prompt_template, config, None
        
    except Exception as e:
        return None, None, None, None, str(e)


# Initialize system
retriever, llm_handler, prompt_template, config, error = load_rag_system()

# Header
st.title("TP-NLP RAG chatbot")

if error:
    st.error(f"rrror loading system: {error}")
    st.info("Make sure you've indexed documents first")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("ssources"):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"{i}. {Path(source['source']).name} (Page {source['page']}) - Score: {source['score']:.3f}")

# Chat input
if prompt := st.chat_input("Ask about docs..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner(".."):
            try:
                result = llm_handler.answer_with_retrieval(
                    question=prompt,
                    retriever=retriever,
                    prompt_template=prompt_template,
                    top_k=config['retrieval']['top_k']
                )
                
                st.markdown(result['answer'])
                
                # Show sources
                with st.expander("sources"):
                    for i, source in enumerate(result['sources'], 1):
                        st.caption(f"{i}. {Path(source['source']).name} (Page {source['page']}) - Score: {source['score']:.3f}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources']
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
