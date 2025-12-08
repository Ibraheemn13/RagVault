import os
import uuid
from typing import List
import shutil

import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

import google.generativeai as genai
from pypdf import PdfReader
import docx  # python-docx


# ============ CONFIG: API KEY & PATHS ============

# 1) GEMINI API KEY
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.set_page_config(page_title="RAG with Gemini & Streamlit")
    st.error("GEMINI_API_KEY is not set. Set it as an env var or in the code.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# 2) Paths for storage
UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


# ============ EMBEDDING FUNCTION (GEMINI) ============

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Chroma embedding function wrapper around Google Gemini embeddings.
    Must implement __call__ and name() for Chroma 0.5+.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for t in texts:
            res = genai.embed_content(model=self.model_name, content=t)
            embeddings.append(res["embedding"])
        return embeddings

    def name(self) -> str:
        # Any stable name string is fine; this is stored in Chroma's metadata
        return f"gemini-{self.model_name}"


embedding_fn = GeminiEmbeddingFunction()


# ============ CHROMA SETUP ============

def get_chroma_collection():
    """
    Create/load a persistent Chroma collection using Gemini embeddings.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="rag_documents_v2",
        embedding_function=embedding_fn
    )
    return collection


collection = get_chroma_collection()


# ============ FILE PARSING HELPERS ============

def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text


def load_docx_text(file_path: str) -> str:
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def load_file_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return load_pdf_text(file_path)
    elif file_path.lower().endswith(".docx"):
        return load_docx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# ============ TEXT CHUNKING ============

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Simple word-based chunking.
    chunk_size/overlap here are in *words*, not characters.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c.strip()]


# ============ INGESTION PIPELINE ============

def ingest_file_into_chroma(file_path: str):
    """
    Load a PDF/DOCX, chunk its text, and add to Chroma collection.
    """
    file_name = os.path.basename(file_path)
    st.write(f"Processing: `{file_name}` ...")

    raw_text = load_file_text(file_path)
    if not raw_text.strip():
        st.warning(f"No text extracted from {file_name}. Skipping.")
        return

    chunks = chunk_text(raw_text)
    st.write(f"Extracted {len(chunks)} chunks from `{file_name}`.")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": file_name} for _ in chunks]

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    st.success(f"Ingested `{file_name}` into vector store.")


def ingest_uploaded_files(uploaded_files):
    """
    Save uploaded files to disk and ingest them into Chroma.
    """
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return

    for uploaded_file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # Save file to disk
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ingest into vector store
        try:
            ingest_file_into_chroma(save_path)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


# ============ RAG: RETRIEVE + GENERATE ============

def retrieve_relevant_chunks(query: str, k: int = 5):
    """
    Query Chroma to get the top-k relevant chunks for a user question.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    # results["documents"] is List[List[str]], we only have one query, so [0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return list(zip(docs, metadatas))


def build_prompt_with_context(question: str, docs_and_meta) -> str:
    """
    Build a prompt for Gemini that includes retrieved context.
    """
    context_blocks = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        source = meta.get("source", "unknown")
        context_blocks.append(f"[Chunk {i} | Source: {source}]\n{doc}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    prompt = f"""
You are a helpful assistant answering questions based only on the provided context.

CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
- If the context does not contain the answer, say you do not know or that the documents do not mention it.
- Cite the chunk numbers or source file names when useful.
- Answer clearly and concisely.
"""
    return prompt


def answer_question_with_rag(question: str) -> str:
    docs_and_meta = retrieve_relevant_chunks(question, k=5)
    prompt = build_prompt_with_context(question, docs_and_meta)

    # Updated model name:
    model = genai.GenerativeModel("gemini-flash-latest")
    # safe fallback:
    # model = genai.GenerativeModel("gemini-pro")

    response = model.generate_content(prompt)
    return response.text


# ============ STREAMLIT UI ============

st.set_page_config(page_title="RAG over PDFs/DOCX with Gemini", layout="wide")

# Basic custom styling
st.markdown(
    """
    <style>
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .sidebar-section-title {
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    .sidebar-caption {
        font-size: 0.8rem;
        color: #6b7280;
    }
    .stat-block {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        background-color: #f9fafb;
    }
    .stat-label {
        color: #6b7280;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.15rem;
    }
    .stat-value {
        color: #111827;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">RAG Vault: Chat over Your Documents</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload PDF or DOCX files in the sidebar, ingest them into a vector store, and then chat with a Gemini-powered assistant grounded in your documents.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">Upload and ingest documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Select PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if st.button("Upload files into vector store"):
        ingest_uploaded_files(uploaded_files)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Storage</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-caption">Vector store directory:<br><code>{CHROMA_DIR}</code></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-caption">Uploaded files directory:<br><code>{UPLOAD_DIR}</code></div>', unsafe_allow_html=True)

    # Show simple stats
    try:
        num_chunks = collection.count()
    except Exception:
        num_chunks = "?"

    st.markdown(
        f"""
        <div class="stat-block">
            <div class="stat-label">Indexed chunks</div>
            <div class="stat-value">{num_chunks}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">Maintenance</div>', unsafe_allow_html=True)
    if st.button("Clear vector store (chroma_db)"):
        try:
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            os.makedirs(CHROMA_DIR, exist_ok=True)
            # Recreate collection so the app continues to work without reload
            collection = get_chroma_collection()
            st.success("Vector store cleared. It will be recreated on next use.")
        except Exception as e:
            st.error(f"Error clearing vector store: {e}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Documents and embeddings are stored on this server only. Clearing the vector store will remove all indexed chunks.</div>',
        unsafe_allow_html=True,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Chat with your documents")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about your uploaded documents...")
if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response using RAG
    with st.chat_message("assistant"):
        with st.spinner("Generating answer using RAG..."):
            try:
                answer = answer_question_with_rag(user_input)
            except Exception as e:
                answer = f"Error during RAG answer: {e}"
        st.markdown(answer)

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
