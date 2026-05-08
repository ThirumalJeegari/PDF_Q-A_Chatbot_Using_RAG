import streamlit as st
from dotenv import load_dotenv
import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG PDF Q&A Chatbot")

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API Key not found")
    st.stop()

# =====================================================
# SESSION STATE
# =====================================================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload PDF",
    type="pdf"
)

# =====================================================
# PROCESS PDF BUTTON
# =====================================================
if uploaded_file:

    if st.button("📥 Submit PDF"):

        with st.spinner("📖 Reading and Processing PDF..."):

            # Save PDF
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            # Load PDF
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()

            # Split Text
            splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(documents)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # FAISS DB
            db = FAISS.from_documents(
                docs,
                embeddings
            )

            st.session_state.vector_db = db

            # =================================================
            # CREATE SUMMARY
            # =================================================
            context = "\n".join(
                [doc.page_content for doc in docs[:5]]
            )

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                groq_api_key=GROQ_API_KEY
            )

            summary_prompt = f"""
            Give a clean summary of this PDF.

            Context:
            {context}
            """

            summary_response = llm.invoke(summary_prompt)

            st.session_state.summary = summary_response.content

        st.success("✅ PDF Processed Successfully!")

# =====================================================
# SHOW SUMMARY
# =====================================================
if st.session_state.summary != "":

    st.subheader("📝 PDF Summary")

    st.write(st.session_state.summary)

# =====================================================
# QUESTION SECTION
# =====================================================
if st.session_state.vector_db:

    st.divider()

    st.subheader("💬 Ask Questions From PDF")

    query = st.text_input(
        "Ask your question"
    )

    # =================================================
    # ASK BUTTON
    # =================================================
    if st.button("🚀 Ask Question"):

        if query.strip() != "":

            with st.spinner("🤖 Thinking..."):

                try:

                    # Similarity Search
                    retrieved_docs = (
                        st.session_state.vector_db
                        .similarity_search(query)
                    )

                    # Context
                    context = "\n".join(
                        [
                            doc.page_content
                            for doc in retrieved_docs
                        ]
                    )

                    # LLM
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        groq_api_key=GROQ_API_KEY
                    )

                    # Prompt
                    prompt = f"""
                    Answer the question using only the
                    provided context.

                    Context:
                    {context}

                    Question:
                    {query}
                    """

                    # Response
                    response = llm.invoke(prompt)

                    answer = response.content

                    # Store History
                    st.session_state.chat_history.append(
                        {
                            "question": query,
                            "answer": answer
                        }
                    )

                except Exception as e:

                    st.error(f"❌ Error: {str(e)}")

# =====================================================
# CHAT HISTORY
# =====================================================
if st.session_state.chat_history:

    st.divider()

    st.subheader("📚 Conversation History")

    for i, chat in enumerate(
        st.session_state.chat_history
    ):

        st.markdown(
            f"""
            ### 🧑 Question {i+1}
            {chat['question']}

            ### 🤖 Answer
            {chat['answer']}
            """
        )

        st.divider()

# =====================================================
# END CHAT
# =====================================================
if st.session_state.vector_db:

    if st.button("❌ End Chat"):

        st.session_state.vector_db = None
        st.session_state.chat_history = []
        st.session_state.summary = ""

        st.success("Chat Ended Successfully!")

        st.rerun()
