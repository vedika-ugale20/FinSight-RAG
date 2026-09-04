# FinSight RAG
### Enterprise Financial Document Intelligence & Retrieval-Augmented Generation Assistant

FinSight RAG is an enterprise financial document question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions from Microsoft's 2023, 2024, and 2025 annual reports.

The system retrieves relevant evidence from the financial documents before generating an answer, helping reduce unsupported responses and improving source transparency.

---

## 📌 Project Overview

Financial annual reports contain large amounts of structured and unstructured information. Finding specific financial information manually can be time-consuming, especially when comparing multiple fiscal years.

FinSight RAG addresses this problem by allowing users to ask natural-language questions about Microsoft's annual reports and receive answers supported by retrieved document evidence.

The system combines:

- Document processing
- Metadata-aware text chunking
- Sentence Transformer embeddings
- FAISS vector similarity search
- Evidence-based retrieval
- FLAN-T5 grounded generation
- Fiscal-year filtering
- Source attribution
- Insufficient-evidence handling
- Interactive Streamlit dashboard

---

## 🎯 Objectives

The main objectives of FinSight RAG are:

1. Build a financial document question-answering system using RAG.
2. Process Microsoft's annual reports from multiple fiscal years.
3. Retrieve relevant financial information using semantic similarity.
4. Generate answers using retrieved document evidence.
5. Provide source attribution for retrieved information.
6. Support fiscal-year based document filtering.
7. Reduce hallucinated or unsupported answers.
8. Provide an interactive interface through Streamlit.

---

## 📄 Documents Used

The system uses Microsoft's annual reports for:

| Fiscal Year | Document |
|-------------|----------|
| 2023 | Microsoft 2023 Annual Report |
| 2024 | Microsoft 2024 Annual Report |
| 2025 | Microsoft 2025 Annual Report |

The documents are processed into smaller text chunks and enriched with metadata such as document name and fiscal year.

---

## 🏗️ System Architecture

```text
Microsoft Annual Reports
          ↓
Document Extraction
          ↓
Text Cleaning
          ↓
Metadata-Aware Chunking
          ↓
Sentence Transformer
(All-MiniLM-L6-v2)
          ↓
Normalized Embeddings
          ↓
FAISS Vector Index
          ↓
Semantic Retrieval
          ↓
Fiscal-Year Filtering
          ↓
Evidence Verification
          ↓
FLAN-T5 Generation
          ↓
Grounded Answer
          ↓
Source Attribution
          ↓
Streamlit Dashboard
