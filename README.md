# FinSight RAG 📊

## Enterprise Financial Document Intelligence & RAG Assistant

FinSight is a Retrieval-Augmented Generation (RAG) system designed to answer financial questions using Microsoft's 2023–2025 annual reports.

### 🚀 Features

- Semantic document retrieval
- FAISS vector database
- 384-dimensional Sentence Transformer embeddings
- Metadata-aware document chunks
- Fiscal-year filtering
- Grounded financial answers
- Source attribution and evidence display
- Hallucination control
- Streamlit interactive dashboard
- FLAN-T5 based grounded generation

### 🧠 Technology Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- NumPy
- Pandas

### 📄 Documents

The system uses Microsoft annual reports for:

- FY2023
- FY2024
- FY2025

### 🔄 RAG Pipeline

Documents → Text Extraction → Chunking → Embeddings → FAISS Retrieval → Evidence Verification → Grounded Answer

### 💡 Example Questions

- What was Microsoft's revenue in fiscal year 2024?
- What was Microsoft's net income in 2025?
- How did Microsoft's revenue change from 2023 to 2025?
- What was Microsoft's operating income in 2024?

### 📊 Project Statistics

- 3 annual reports
- 246 metadata-aware chunks
- 384-dimensional embeddings
- FAISS similarity search

### ⚠️ Scope

FinSight answers questions only from the provided Microsoft annual reports. It does not use external financial information.
