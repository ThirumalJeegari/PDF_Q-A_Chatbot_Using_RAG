import streamlit as st
from dotenv import load_dotenv
import os


# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Check API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("RAG PDF Q&A Chatbot")

if not GROQ_API_KEY:
    st.error("GROQ API Key not found. Please check your .env file")
    st.stop()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Create embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in FAISS
    db = FAISS.from_documents(docs, embeddings)

    st.success("Document processed successfully!")

    # Question input
    query = st.text_input("Ask a question from the document:")

    if query:
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant chunks
                retrieved_docs = db.similarity_search(query)

                # Combine context
                context = "\n".join([doc.page_content for doc in retrieved_docs])

                # Groq LLM
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    groq_api_key=GROQ_API_KEY
                )

                # Prompt
                prompt = f"""
                Answer the question based only on the context below.

                Context:
                {context}

                Question:
                {query}
                """

                # Generate response
                response = llm.invoke(prompt)

                st.write("Your Answer:")
                st.write(response.content)

            except Exception as e:
                st.error(f"Error: {str(e)}")