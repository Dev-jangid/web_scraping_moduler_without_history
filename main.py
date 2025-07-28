import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, get_groq_client, VECTOR_STORE_DIR
from web_utils import fetch_website_content, process_content
from vector_utils import vector_store_exists, create_vector_store, load_vector_store, retrieve_context
from groq_utils import generate_chat_response

# Initialize Groq client
try:
    client = get_groq_client()
except Exception as e:
    st.error(str(e))
    st.stop()

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)
embedding_model = load_embedding_model()

# Initialize session state
if 'current_url' not in st.session_state:
    st.session_state.current_url = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None

# Streamlit app layout
st.set_page_config(
    page_title="WebChat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - URL input and vector management
with st.sidebar:
    st.title("💬 WebChat Assistant")
    
    # URL input form
    with st.form("url_form", clear_on_submit=False):
        url = st.text_input("Enter Website URL:", 
                          placeholder="https://example.com")
        submit_url = st.form_submit_button("Process Website")
        
        if submit_url and url:
            if not url.startswith('http'):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Processing website..."):
                    try:
                        # Fetch and process content
                        raw_text = fetch_website_content(url)
                        if not raw_text:
                            st.error("Failed to fetch content from URL")
                        else:
                            processed_text = process_content(raw_text)
                            
                            # Check for existing vector store
                            if vector_store_exists(url):
                                st.info("Loading existing vector database")
                                vector_store, chunks = load_vector_store(url)
                            else:
                                st.info("Creating new vector database")
                                vector_store, chunks = create_vector_store(processed_text, embedding_model, url)
                            
                            # Store in session state
                            st.session_state.current_url = url
                            st.session_state.vector_store = vector_store
                            st.session_state.chunks = chunks
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Vector Database")
    
    # Show current status
    if st.session_state.current_url:
        st.info(f"Loaded: {st.session_state.current_url[:50]}...")
    else:
        st.info("No website loaded")
    
    # Clear databases button
    if st.button("Clear All Vector Databases"):
        for file in VECTOR_STORE_DIR.glob("*"):
            file.unlink()
        st.session_state.current_url = ""
        st.session_state.vector_store = None
        st.session_state.chunks = None
        st.success("All vector databases cleared")
        st.rerun()

# Main Chat Area
st.title("WebChat Assistant")
st.caption("Chat with any website using AI")

if st.session_state.current_url:
    # Q&A interface
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", 
                                 placeholder="Ask about this website...")
        submit_chat = st.form_submit_button("Ask")
        
        if submit_chat and user_input:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve context
                    context = retrieve_context(
                        user_input, 
                        st.session_state.vector_store, 
                        st.session_state.chunks, 
                        embedding_model
                    )
                    
                    # Generate response
                    bot_response = generate_chat_response(
                        client,
                        user_input, 
                        context
                    )
                    
                    # Display response
                    st.subheader("Answer")
                    st.write(bot_response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👈 Enter a website URL in the sidebar to get started")

st.divider()
st.caption("WebChat Assistant v1.3 | Single URL mode | Local vector database")



