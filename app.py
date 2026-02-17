import streamlit as st
import os
from dotenv import load_dotenv
from rag_backend import MedicalRAGBackend

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Report Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏥 Medical Report Analyzer")
st.write(
    "Upload your medical report in PDF format. "
    "The system will analyze and provide a simplified, evidence-based explanation using RAG."
)

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input (optional now)
    api_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Leave empty for free mode. With key: AI-powered explanations. Without: extract key information from report."
    )
    
    if not api_key:
        st.info("🆓 Running in FREE mode - Uses extractive summarization (no API needed)")
    else:
        st.success("✅ OpenAI mode enabled")
    
    # Query customization
    use_custom_query = st.checkbox("Use custom explanation focus", value=False)
    custom_query = ""
    if use_custom_query:
        custom_query = st.text_area(
            "What would you like to focus on?",
            placeholder="e.g., 'Explain the test results and their implications'",
            height=100
        )
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider("Number of relevant chunks to retrieve", 1, 10, 3)
    chunk_size = st.slider("Chunk size (characters)", 300, 1000, 500, step=100)
    overlap = st.slider("Chunk overlap (characters)", 0, 200, 100, step=50)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Medical Report")
    uploaded_file = st.file_uploader(
        "Select PDF file",
        type=["pdf"],
        help="Upload your medical report in PDF format"
    )

with col2:
    st.subheader("📊 Processing Status")
    status_placeholder = st.empty()

st.divider()

# Analyze button
analyze_button = st.button(
    "🔍 Analyze Report",
    type="primary",
    use_container_width=True
)

st.divider()

# Result section
if analyze_button:
    # Validation
    if uploaded_file is None:
        st.error("❌ Please upload a PDF file first.")
    else:
        try:
            # Update status
            status_placeholder.info("🔄 Initializing RAG pipeline...")
            
            # Initialize backend
            backend = MedicalRAGBackend(api_key=api_key if api_key else None)
            
            # Step 1: Extract text
            status_placeholder.info("📖 Extracting text from PDF...")
            raw_text = backend.extract_text_from_pdf(uploaded_file)
            
            if len(raw_text.strip()) == 0:
                st.error("❌ Could not extract text from PDF. Please ensure the PDF contains readable text.")
            else:
                # Step 2: Chunk text
                status_placeholder.info("✂️ Splitting text into chunks...")
                chunks = backend.chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
                st.success(f"✅ Created {len(chunks)} text chunks")
                
                # Step 3: Create embeddings
                status_placeholder.info("🧠 Creating embeddings...")
                embeddings = backend.create_embeddings(chunks)
                
                # Step 4: Build vector store
                status_placeholder.info("💾 Building vector database...")
                backend.build_vector_store(embeddings)
                
                # Step 5: Retrieve relevant chunks
                status_placeholder.info("🔎 Retrieving relevant information...")
                query = custom_query or "Summarize and explain the key findings in this medical report"
                relevant_chunks = backend.retrieve_relevant_chunks(query, top_k=top_k)
                
                # Step 6: Generate explanation
                status_placeholder.info("✍️ Generating medical explanation...")
                explanation = backend.generate_explanation(query, relevant_chunks)
                
                # Clear status
                status_placeholder.success("✅ Analysis complete!")
                
                # Display results
                st.subheader("📋 Medical Explanation")
                st.markdown(explanation)
                
                # Display source chunks
                with st.expander("📚 Source Information (Retrieved Chunks)"):
                    for i, chunk in enumerate(relevant_chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.text_area(
                            f"Source chunk {i}",
                            value=chunk,
                            height=100,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"chunk_{i}"
                        )
                        st.divider()
                
                # Statistics
                with st.expander("📊 Processing Statistics"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Chunks", len(chunks))
                    col2.metric("Retrieved Chunks", len(relevant_chunks))
                    col3.metric("Raw Text Length", f"{len(raw_text):,} chars")
                    col4.metric("Embedding Dimension", backend.embedding_dim)
        
        except Exception as e:
            status_placeholder.error(f"❌ Error during processing")
            st.error(f"Error details: {str(e)}")
            if "API" in str(e).upper():
                st.info("💡 This error is related to OpenAI API. Try again or run in free mode without an API key.")

# Disclaimer
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only and does not provide medical advice. "
    "Always consult with a qualified healthcare professional for accurate diagnosis and treatment."
)
