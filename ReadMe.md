# 📄 PDF Q&A Chatbot (RAG + Groq)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG)** based AI chatbot.  
It allows users to upload PDF documents and ask questions.  
The system extracts relevant content and generates accurate answers using **Groq LLM**.

---

## 🎯 Features

✔ Upload PDF documents  
✔ Ask questions from PDF  
✔ Context-aware AI responses  
✔ Fast inference using Groq  
✔ Semantic search using FAISS  
✔ Simple Streamlit UI  

---

## 🧠 Tech Stack

- Python  
- Streamlit  
- LangChain  
- FAISS  
- Groq API  
- HuggingFace Embeddings  
- PyPDF  

---

## 📁 Project Structure

rag-pdf-chatbot/
│

├── app

├── requirements.txt

├── .env.example

├── .gitignore

└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone [https://github.com/your-username/rag-pdf-chatbot.git ](https://github.com/ThirumalJeegari/PDF-Q-A-Chatbot-Using-RAG.git) 
cd rag-pdf-chatbot  

---

### 2️⃣ Install dependencies
pip install -r requirements.txt  

---

### 3️⃣ Setup environment variables

Create a `.env` file and add:

GROQ_API_KEY=your_groq_api_key_here  

---

## ▶️ Run the Application

streamlit run app/app.py  

---

## 🧠 How It Works

1. Upload PDF 📄  
2. Extract text from document  
3. Split text into chunks  
4. Convert chunks into embeddings  
5. Store embeddings in FAISS  
6. User asks a question ❓  
7. Retrieve relevant chunks  
8. Groq LLM generates answer 🤖  

---


## 🧪 Example Use Cases

📚 Study notes assistant  
📄 Resume analyzer  
🏢 Company policy chatbot  
📖 Research paper Q&A  

---

## 💡 Sample Questions

- What is explained in chapter 2?  
- Summarize this document  
- What are the key points?  

---

## 🚀 Future Improvements

- Chat history memory  
- Multiple PDF support  
- Highlight answers in PDF  
- Cloud deployment  
- ChatGPT-style UI  

---

## 👨‍💻 Author

Jeegari Thirumal
 