# streamlit_app.py
import os
import base64
import tempfile
import streamlit as st
from PIL import Image
from project2 import AIAgent, ModelFactory

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #424242;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F5F5F5;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        height: 60px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .chat-message-user {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #1E88E5;
        color: #000000 !important;
    }
    .chat-message-ai {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #424242;
        color: #000000 !important;
    }
    .content-box {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1E88E5;
        font-weight: 600;
    }
    /* Fix for dark mode text visibility - comprehensive version */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    .stTextInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* Ensure all markdown text is visible */
    .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown h1, 
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    .stMarkdown li, .stMarkdown a, .stMarkdown td, .stMarkdown th {
        color: #000000 !important;
    }
    /* Fix for all info/warning/error/success boxes */
    .stAlert p, .stInfo p, .stWarning p, .stError p, .stSuccess p {
        color: #000000 !important;
    }
    /* Ensure select boxes and radio buttons have visible text */
    .stSelectbox label, .stRadio label {
        color: #000000 !important;
    }
    /* Fix text in expanders */
    .streamlit-expanderHeader, .streamlit-expanderContent {
        color: #000000 !important;
    }
    /* Fix for metrics */
    .stMetric label, .stMetric div {
        color: #000000 !important;
    }
    /* Ensure chat bubbles have visible text */
    .chat-container p, .chat-container strong, .chat-container span, 
    .chat-container div, .chat-container h1, .chat-container h2, 
    .chat-container h3, .chat-container h4, .chat-container h5, 
    .chat-container h6, .chat-container li, .chat-container a {
        color: #000000 !important;
    }
    /* Fix for all sidebar elements */
    .sidebar .stMarkdown p, .sidebar .stMarkdown span, .sidebar .stMarkdown div,
    .sidebar .stButton, .sidebar .stSelectbox, .sidebar .stCheckbox,
    .sidebar .stRadio, .sidebar .stTextInput, .sidebar .stTextArea,
    .sidebar .stExpander, .sidebar .stInfo, .sidebar .stWarning,
    .sidebar .stError, .sidebar .stSuccess {
        color: #000000 !important;
    }
    /* Additional fixes for any other text-containing elements */
    .stText, .stProgress, .stProgress > div, .stProgress > div > div {
        color: #000000 !important;
    }
    /* Ensure text in tabs is visible */
    .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
    }
    /* Fix text in file uploader */
    .stFileUploader label, .stFileUploader span, .stFileUploader p {
        color: #000000 !important;
    }
    /* Ensure text in charts and dataframes is visible */
    .stDataFrame div, .stDataFrame text {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Document Understanding AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "processed_items" not in st.session_state:
    st.session_state.processed_items = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Title and description with styled elements
st.markdown('<p class="main-header">📄 Document Understanding AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents, images, or provide URLs to analyze and query content.</p>', unsafe_allow_html=True)

# Sidebar with styled elements
st.sidebar.markdown('<p class="sidebar-header">Model Configuration</p>', unsafe_allow_html=True)

# Add a cool animation for the sidebar
# st.sidebar.markdown("""
# <div style="display: flex; justify-content: center; margin-bottom: 20px;">
#     <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #1E88E5; border-radius: 50%; animation: spin 2s linear infinite;"></div>
# </div>
# <style>
# @keyframes spin {
#     0% { transform: rotate(0deg); }
#     100% { transform: rotate(360deg); }
# }
# </style>
# """, unsafe_allow_html=True)

model_type = st.sidebar.selectbox(
    "Select AI Model",
    ["gemini", "anthropic", "openai"],
    index=0
)

api_key = st.sidebar.text_input(
    "API Key",
    value="AIzaSyA03TDmMCySeHgisstcCLeBurY9NnyCytE" if model_type == "gemini" else "",
    type="password"
)

model_names = {
    "gemini": ["gemini-2.5-pro", "gemini-1.5-pro"],
    "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4"]
}

model_name = st.sidebar.selectbox(
    "Model Name",
    model_names[model_type]
)

# Add verification model configuration to sidebar with collapsible section
st.sidebar.markdown('<p class="sidebar-header">Verification Model</p>', unsafe_allow_html=True)
use_separate_verification = st.sidebar.checkbox("Use separate model for verification")

verification_model_config = None
if use_separate_verification:
    with st.sidebar.expander("Verification Model Settings", expanded=True):
        verification_model_type = st.selectbox(
            "Select Verification Model",
            ["gemini", "anthropic", "openai"],
            index=0,
            key="verification_model_type"
        )
        
        verification_api_key = st.text_input(
            "Verification API Key",
            value="AIzaSyA03TDmMCySeHgisstcCLeBurY9NnyCytE" if verification_model_type == "gemini" else "",
            type="password",
            key="verification_api_key"
        )
        
        verification_model_name = st.selectbox(
            "Verification Model Name",
            model_names[verification_model_type],
            key="verification_model_name"
        )
        
        verification_model_config = {
            "type": verification_model_type,
            "api_key": verification_api_key,
            "model_name": verification_model_name
        }

# Update the agent initialization with progress indicator
if st.sidebar.button("Initialize Agent", key="init_button") or st.session_state.agent is None:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        if i < 30:
            status_text.text("Loading models...")
        elif i < 70:
            status_text.text("Configuring environment...")
        else:
            status_text.text("Finalizing initialization...")
        if i < 100:
            import time
            time.sleep(0.01)
    
    model_config = {
        "type": model_type,
        "api_key": api_key,
        "model_name": model_name
    }
    st.session_state.agent = AIAgent(model_config, verification_model_config)
    status_text.empty()
    progress_bar.empty()
    st.sidebar.success("✅ Agent initialized successfully!")

# Main content area - tabs for different functions with styled tabs
tab1, tab2 = st.tabs(["📥 Process Content", "💬 Chat with Documents"])

# Tab 1: Process Content
with tab1:
    st.markdown('<h2 style="color: #1E88E5;">Upload or Link Content</h2>', unsafe_allow_html=True)
    
    # Input options with more visual elements
    input_method = st.radio(
        "Select input method:",
        ["Upload Files", "Enter URL", "Enter Local Path"],
        horizontal=True
    )
    
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    
    if input_method == "Upload Files":
        st.markdown("### 📂 Upload Files")
        uploaded_files = st.file_uploader(
            "Drag and drop documents or images",
            type=["pdf", "jpg", "jpeg", "png", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            file_count = len(uploaded_files)
            st.info(f"Ready to process {file_count} file{'s' if file_count > 1 else ''}")
            
            if st.button("🚀 Process Uploads", key="process_uploads"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = int((i / len(uploaded_files)) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Process the file
                    new_items = st.session_state.agent.process_file(temp_path)
                    
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                
                progress_bar.progress(100)
                status_text.empty()
                st.session_state.processing_complete = True
                st.success(f"✅ Processing complete! Found {len(st.session_state.processed_items)} content items")
    
    elif input_method == "Enter URL":
        st.markdown("### 🔗 Enter URL")
        url = st.text_input("Enter URL to process:", placeholder="https://example.com")
        
        if url:
            if st.button("🚀 Process URL", key="process_url"):
                with st.spinner("Processing URL..."):
                    status = st.empty()
                    status.text("Connecting to URL...")
                    
                    progress_bar = st.progress(25)
                    status.text("Downloading content...")
                    
                    progress_bar.progress(50)
                    status.text("Analyzing content...")
                    
                    progress_bar.progress(75)
                    status.text("Extracting information...")
                    
                    new_items = st.session_state.agent.process_url(url)
                    
                    progress_bar.progress(100)
                    status.empty()
                    
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)
                        st.session_state.processing_complete = True
                        st.success(f"✅ Processed URL: Found {len(new_items)} content items")
                    else:
                        st.error("❌ Failed to process URL")
    
    elif input_method == "Enter Local Path":
        st.markdown("### 📁 Enter Local Path")
        local_path = st.text_input("Enter local file or directory path:", placeholder="C:/path/to/file.pdf")
        
        if local_path:
            if st.button("🚀 Process Path", key="process_path"):
                with st.spinner("Processing path..."):
                    status = st.empty()
                    progress_bar = st.progress(0)
                    
                    for i in range(5):
                        progress_bar.progress(i * 20)
                        status.text(f"Processing step {i+1}/5...")
                        import time
                        time.sleep(0.2)
                    
                    new_items = st.session_state.agent.process_input(local_path)
                    
                    progress_bar.progress(100)
                    status.empty()
                    
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)
                        st.session_state.processing_complete = True
                        st.success(f"✅ Processed path: Found {len(new_items)} content items")
                    else:
                        st.error("❌ Failed to process path")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display processed items summary with interactive elements
    if st.session_state.processed_items:
        st.markdown('<h2 style="color: #1E88E5;">Processed Content</h2>', unsafe_allow_html=True)
        
        # Create summary with metrics
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("📊 Content Summary")
        
        # Summarize content types
        content_types = {}
        for item in st.session_state.processed_items:
            content_types[item.type] = content_types.get(item.type, 0) + 1
        
        # Display metrics in columns
        cols = st.columns(len(content_types) if content_types else 1)
        for i, (content_type, count) in enumerate(content_types.items()):
            with cols[i]:
                st.metric(f"{content_type.capitalize()}", count)
        
        # Special content identified in images
        special_content = []
        for item in st.session_state.processed_items:
            if item.type == "image" and "identified_content" in item.metadata:
                for content_type in item.metadata["identified_content"]:
                    if content_type not in ["image", "photo"]:
                        special_content.append(content_type)
        
        if special_content:
            st.markdown("### 🔍 Special Content Identified")
            st.markdown(", ".join([f"**{content}**" for content in set(special_content)]))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show samples of processed items with tabs
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 📋 Content Samples")
        
        content_tabs = st.tabs(["Text", "Tables", "Images", "Pages"])
        
        with content_tabs[0]:  # Text
            text_items = [item for item in st.session_state.processed_items if item.type == "text"][:5]
            if text_items:
                for i, item in enumerate(text_items):
                    with st.expander(f"Text from {os.path.basename(item.path)}", expanded=i==0):
                        st.text_area("Content", item.content, height=150, key=f"text_{i}")
            else:
                st.info("No text content available")
                
        with content_tabs[1]:  # Tables
            table_items = [item for item in st.session_state.processed_items if item.type == "table"][:5]
            if table_items:
                for i, item in enumerate(table_items):
                    with st.expander(f"Table from {os.path.basename(item.path)}", expanded=i==0):
                        st.text_area("Content", item.content, height=150, key=f"table_{i}")
            else:
                st.info("No table content available")
                
        with content_tabs[2]:  # Images
            image_items = [item for item in st.session_state.processed_items if item.type == "image"][:5]
            if image_items:
                for i, item in enumerate(image_items):
                    try:
                        st.image(item.path, caption=f"{os.path.basename(item.path)}", use_column_width=True)
                    except:
                        st.error(f"Could not display image: {item.path}")
            else:
                st.info("No image content available")
                
        with content_tabs[3]:  # Pages
            page_items = [item for item in st.session_state.processed_items if item.type == "page"][:5]
            if page_items:
                for i, item in enumerate(page_items):
                    try:
                        st.image(item.path, caption=f"{os.path.basename(item.path)}", use_column_width=True)
                    except:
                        st.error(f"Could not display image: {item.path}")
            else:
                st.info("No page content available")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Chat with Documents
with tab2:
    st.markdown('<h2 style="color: #1E88E5;">Ask Questions About Your Documents</h2>', unsafe_allow_html=True)
    
    if not st.session_state.processed_items:
        st.warning("⚠️ Please process some content first in the 'Process Content' tab.")
    else:
        # Chat input with nice styling
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        
        user_query = st.text_input("Ask a question about your documents:", placeholder="What are the main topics in these documents?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("🔍 Ask", key="submit_question")
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.experimental_rerun()
        
        if user_query and submit_button:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Show typing animation
            with st.spinner(""):
                message_placeholder = st.empty()
                for i in range(5):
                    message_placeholder.markdown("AI is thinking" + "." * (i % 4), unsafe_allow_html=True)
                    import time
                    time.sleep(0.3)
                
                # Get AI response
                response = st.session_state.agent.query_content(user_query)
                message_placeholder.empty()
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display chat history with modern styling
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.info("Your conversation will appear here after you ask a question.")
        else:
            st.markdown('<h3 style="color: #1E88E5; margin-top: 30px;">Conversation</h3>', unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user"><strong style="color:#0D47A1;">You:</strong> <span style="color:#000000;">{message["content"]}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-ai"><strong style="color:#424242;">AI:</strong> <span style="color:#000000;">{message["content"]}</span></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with more information and links
st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    This application uses AI models to analyze documents, extract content, and answer questions about the content.
    
    **Features:**
    - Process multiple document types (PDF, images, text)
    - Extract text, tables, and visual content
    - Ask questions about your documents
    - Use verification models for improved accuracy
    """)

st.sidebar.markdown("---")
st.sidebar.info("Powered by AI • Made with ❤️")
