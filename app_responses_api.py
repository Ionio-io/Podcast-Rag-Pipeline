import streamlit as st
import os
from rag_system_responses_api import ResponsesAPIRAG
import tempfile
import locale
import sys

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    # Windows-specific encoding fix
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
else:
    # Unix/macOS encoding fix
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set page config
st.set_page_config(
    page_title="OpenAI Responses API RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag(api_key: str):
    """Initialize the RAG system with the OpenAI API key."""
    return ResponsesAPIRAG(api_key)

def process_uploaded_files(uploaded_files, temp_dir):
    """Process uploaded files and save them to a temporary directory with proper encoding."""
    for uploaded_file in uploaded_files:
        # Ensure filename is safe for filesystem
        try:
            safe_filename = uploaded_file.name.encode('utf-8').decode('utf-8')
        except UnicodeError:
            safe_filename = uploaded_file.name.encode('ascii', errors='ignore').decode('ascii')
        
        file_path = os.path.join(temp_dir, safe_filename)
        
        try:
            # Handle file content with proper encoding
            file_content = uploaded_file.getbuffer()
            with open(file_path, 'wb') as f:
                f.write(file_content)
        except Exception as e:
            st.error(f"Error saving file {safe_filename}: {str(e)}")
            continue
    
    return temp_dir

# Sidebar for API key and file upload
with st.sidebar:
    st.title("🤖 Responses API Settings")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        # Initialize RAG system for listing vector stores
        if 'temp_rag' not in st.session_state:
            st.session_state.temp_rag = initialize_rag(api_key)
        
        # Vector Store Selection
        st.subheader("📁 Vector Store Options")
        
        # Option to use existing vector store
        use_existing = st.radio(
            "Choose option:",
            ["Create New Vector Store", "Use Existing Vector Store"],
            key="vector_store_option"
        )
        
        if use_existing == "Use Existing Vector Store":
            # List existing vector stores
            try:
                vector_stores = st.session_state.temp_rag.list_vector_stores()
                if vector_stores:
                    vector_store_options = {
                        f"{vs['name']} ({vs['file_count']} files)": vs['id'] 
                        for vs in vector_stores
                    }
                    
                    selected_vs_name = st.selectbox(
                        "Select Vector Store:",
                        options=list(vector_store_options.keys()),
                        key="selected_vector_store"
                    )
                    
                    if st.button("🔗 Connect to Vector Store", type="primary"):
                        selected_vs_id = vector_store_options[selected_vs_name]
                        with st.spinner("Connecting to vector store..."):
                            st.session_state.rag = initialize_rag(api_key)
                            if st.session_state.rag.connect_to_vector_store(selected_vs_id):
                                st.success(f"✅ Connected to {selected_vs_name}")
                                st.info(f"📁 Found {len(st.session_state.rag.uploaded_files)} files")
                            else:
                                st.error("❌ Failed to connect to vector store")
                else:
                    st.info("No existing vector stores found.")
                    use_existing = "Create New Vector Store"  # Fall back to create new
            except Exception as e:
                st.error(f"Error loading vector stores: {str(e)}")
                use_existing = "Create New Vector Store"
        
        if use_existing == "Create New Vector Store":
            # File upload
            st.subheader("📤 Upload New Transcripts")
            uploaded_files = st.file_uploader(
                "Upload JSON/TXT transcript files",
                type=['json', 'txt'],
                accept_multiple_files=True,
                help="Upload your transcript files to create a new searchable knowledge base"
            )
            
            if uploaded_files:
                vector_store_name = st.text_input(
                    "Vector Store Name:", 
                    value=f"Transcripts-{len(uploaded_files)}-files",
                    help="Give your vector store a descriptive name"
                )
                
                if st.button("🚀 Create Vector Store", type="primary"):
                    with st.spinner("Creating vector store and processing files..."):
                        try:
                            # Create temporary directory for uploaded files
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save uploaded files with proper encoding
                                process_uploaded_files(uploaded_files, temp_dir)
                                
                                # Initialize RAG system
                                st.session_state.rag = initialize_rag(api_key)
                                
                                # Set custom name for vector store
                                st.session_state.rag.create_vector_store(vector_store_name)
                                
                                # Upload files to vector store
                                for filename in os.listdir(temp_dir):
                                    if filename.endswith(('.json', '.txt')):
                                        file_path = os.path.join(temp_dir, filename)
                                        st.session_state.rag.upload_file(file_path)
                                
                            st.success("✅ Vector store created successfully!")
                            st.info(f"📁 Processed {len(uploaded_files)} files")
                            st.info("🔍 Ready to answer questions using Responses API")
                        except UnicodeEncodeError as unicode_error:
                            st.error(f"❌ Unicode encoding error: {str(unicode_error)}")
                            st.info("💡 This error can occur if your API key contains special characters. Try copying and pasting your API key directly instead of from a rich text source.")
                            st.info("🔧 Also ensure your transcript files contain valid UTF-8 text.")
                        except Exception as e:
                            st.error(f"❌ Error processing files: {str(e)}")
                            st.info("💡 Make sure you have the latest OpenAI client: `pip install openai>=1.54.0`")

    # Action buttons (show if we have a RAG system)
    if st.session_state.rag:
        st.markdown("---")
        st.subheader("🎛️ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ New Conversation", help="Clear chat history"):
                st.session_state.chat_history = []
                st.session_state.rag.new_conversation()
                st.success("Started new conversation!")
                st.rerun()
        
        with col2:
            if st.button("📊 System Info", help="Show system information"):
                st.info(f"📁 Files: {len(st.session_state.rag.uploaded_files)}")
                st.info(f"💬 Messages: {len(st.session_state.chat_history)}")
                if st.session_state.rag.vector_store:
                    st.info(f"🗂️ Vector Store: {st.session_state.rag.vector_store.id[:20]}...")

# Main chat interface
st.title("🤖 OpenAI Responses API RAG System")
st.markdown("**Ask questions about your transcripts using OpenAI's new Responses API with built-in file search!**")

# Information about the system
if not st.session_state.rag:
    st.info("👆 Upload your transcript files in the sidebar and click 'Process Transcripts' to get started!")
    
    with st.expander("ℹ️ About this System"):
        st.markdown("""
        This system uses **OpenAI's Responses API** (released March 2025) with the following features:
        
        - 🔍 **Native File Search**: Uses OpenAI's built-in file search tool
        - 🤖 **GPT-4o Model**: Latest model optimized for tool use
        - 📄 **Automatic Citations**: Sources are automatically referenced
        - 💬 **Conversation Memory**: Maintains context across questions
        - 🔄 **Fallback Mode**: Uses regular chat completions if Responses API isn't available
        - ✅ **Unicode Support**: Handles special characters and emojis properly
        
        **Requirements:**
        - OpenAI API key (make sure to type it directly, not copy from rich text)
        - Latest OpenAI client: `pip install openai>=1.54.0`
        
        **How it works:**
        1. Upload your transcript files (JSON/TXT)
        2. Files are uploaded to OpenAI for file search
        3. Ask questions - the AI searches through your files automatically
        4. Get answers with source citations
        
        **Troubleshooting Unicode Issues:**
        - If you get encoding errors, ensure your API key doesn't contain special characters
        - Make sure transcript files are in UTF-8 format
        - Type your API key directly instead of copying from rich text applications
        
        *Note: This uses the cutting-edge Responses API. If it's not available, the system will fallback to regular chat completions.*
        """)

# Display system status
if st.session_state.rag:
    uploaded_count = len(st.session_state.rag.uploaded_files)
    st.success(f"✅ System ready with {uploaded_count} files")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # Ensure message content is properly displayed
        try:
            content = str(message["content"]).encode('utf-8').decode('utf-8')
            st.write(content)
        except UnicodeError:
            content = str(message["content"]).encode('ascii', errors='ignore').decode('ascii')
            st.write(content)
            st.caption("⚠️ Some characters were removed due to encoding issues")
        
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    try:
                        filename = str(source['filename']).encode('utf-8').decode('utf-8')
                        st.markdown(f"**📄 Source {i+1}: {filename}**")
                    except UnicodeError:
                        filename = str(source['filename']).encode('ascii', errors='ignore').decode('ascii')
                        st.markdown(f"**📄 Source {i+1}: {filename}**")
                    
                    if 'citation' in source and source['citation']:
                        try:
                            citation = str(source['citation']).encode('utf-8').decode('utf-8')
                            st.text(citation)
                        except UnicodeError:
                            citation = str(source['citation']).encode('ascii', errors='ignore').decode('ascii')
                            st.text(citation)
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask a question about your transcripts..."):
    if not st.session_state.rag:
        st.error("❌ Please upload transcripts and provide an API key first!")
    else:
        # Ensure prompt is properly encoded
        try:
            safe_prompt = prompt.encode('utf-8').decode('utf-8')
        except UnicodeError:
            safe_prompt = prompt.encode('ascii', errors='ignore').decode('ascii')
            st.warning("⚠️ Some special characters in your question were removed")
        
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": safe_prompt})
        with st.chat_message("user"):
            st.write(safe_prompt)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching through your transcripts..."):
                try:
                    response = st.session_state.rag.query(safe_prompt)
                    
                    # Display answer with proper encoding
                    try:
                        answer = str(response["answer"]).encode('utf-8').decode('utf-8')
                        st.write(answer)
                    except UnicodeError:
                        answer = str(response["answer"]).encode('ascii', errors='ignore').decode('ascii')
                        st.write(answer)
                        st.caption("⚠️ Some characters were removed due to encoding issues")
                    
                    # Show which API mode was used
                    if "mode" in response:
                        if "Responses API" in response["mode"]:
                            st.success(f"✅ {response['mode']}")
                        elif "Assistants API" in response["mode"]:
                            st.info(f"ℹ️ {response['mode']}")
                        elif "Chat Completions" in response["mode"]:
                            st.warning(f"⚠️ {response['mode']}")
                        elif "Error" in response["mode"]:
                            st.error(f"❌ {response['mode']}")
                    
                    # Show if this was a fallback response
                    if any("Fallback mode" in str(source.get("citation", "")) for source in response.get("sources", [])):
                        st.info("ℹ️ Using fallback mode - Responses API may not be available yet")
                    
                    # Add sources in an expander
                    if response["sources"]:
                        with st.expander(f"📚 View Sources ({len(response['sources'])} found)"):
                            for i, source in enumerate(response["sources"]):
                                try:
                                    filename = str(source['filename']).encode('utf-8').decode('utf-8')
                                    source_type = source.get('type', 'unknown')
                                    score = source.get('score', 0.0)
                                    if score > 0:
                                        st.markdown(f"**📄 Source {i+1}: {filename}** *(type: {source_type}, relevance: {score:.3f})*")
                                    else:
                                        st.markdown(f"**📄 Source {i+1}: {filename}** *(type: {source_type})*")
                                except UnicodeError:
                                    filename = str(source['filename']).encode('ascii', errors='ignore').decode('ascii')
                                    source_type = source.get('type', 'unknown')
                                    score = source.get('score', 0.0)
                                    if score > 0:
                                        st.markdown(f"**📄 Source {i+1}: {filename}** *(type: {source_type}, relevance: {score:.3f})*")
                                    else:
                                        st.markdown(f"**📄 Source {i+1}: {filename}** *(type: {source_type})*")
                                
                                if 'citation' in source and source['citation']:
                                    try:
                                        citation = str(source['citation']).encode('utf-8').decode('utf-8')
                                        if citation.strip():  # Only show if not empty
                                            st.text_area(f"Context from {filename}:", citation, height=120, key=f"citation_{i}_{len(st.session_state.chat_history)}")
                                        else:
                                            st.caption("No context text available")
                                    except UnicodeError:
                                        citation = str(source['citation']).encode('ascii', errors='ignore').decode('ascii')
                                        if citation.strip():
                                            st.text_area(f"Context from {filename}:", citation, height=120, key=f"citation_{i}_{len(st.session_state.chat_history)}")
                                        else:
                                            st.caption("No context text available")
                                else:
                                    st.caption("No citation text available")
                                st.markdown("---")
                    else:
                        # Debug: Show if no sources were found
                        st.info("ℹ️ No sources were extracted from the response. Check terminal for debug information.")
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                except UnicodeEncodeError as unicode_error:
                    st.error(f"❌ Unicode encoding error: {str(unicode_error)}")
                    st.info("💡 This can happen if your API key contains special characters. Try re-entering your API key.")
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
                    st.info("💡 Try updating OpenAI client: `pip install openai>=1.54.0`")

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI's Responses API with File Search | [Learn more](https://cookbook.openai.com/examples/responses_api/responses_example)*") 