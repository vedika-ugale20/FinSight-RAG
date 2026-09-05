# FinSight RAG
# Enterprise Financial Document Intelligence & Retrieval-Augmented Generation Assistant

FinSight RAG is an enterprise financial document question-answering system that uses Retrieval-Augmented Generation (RAG) to answer questions from Microsoft's 2023, 2024, and 2025 annual reports.

The system retrieves relevant evidence from the financial documents before generating an answer, helping reduce unsupported responses and improving source transparency.

---

##  Project Overview

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

# Objectives

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
 # Documents Used

The system uses Microsoft's annual reports for:

| Fiscal Year | Document |
|-------------|----------|
| 2023 | Microsoft 2023 Annual Report |
| 2024 | Microsoft 2024 Annual Report |
| 2025 | Microsoft 2025 Annual Report |

The documents are processed into smaller text chunks and enriched with metadata such as document name and fiscal year.

---

# System Architecture

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
# Architecture

The FinSight RAG system follows an end-to-end Retrieval-Augmented Generation pipeline:

**Microsoft Annual Reports → Document Extraction → Metadata-Aware Chunking → Sentence Transformer Embeddings → FAISS Vector Search → Semantic Retrieval → Evidence Verification → FLAN-T5 Generation → Grounded Answer + Source Attribution**

[View the System Architecture](FinSight_RAG_Architecture_WordStyle.pdf)

# Screenshots

The repository contains screenshots of the deployed FinSight RAG application, including the dashboard, grounded answers, and fiscal-year filtering.

[View Application Screenshots](screenshots/)

#Project Report

[View / Download the Project Report](FinSight_RAG_Project_Report_Final.pdf)

# Live Deployment

[Open FinSight RAG on Streamlit](https://vedika-ugale20-finsight-rag-app-4paiye.streamlit.app)

# Conclusion

FinSight RAG demonstrates an end-to-end financial document question-answering system that combines semantic retrieval, metadata-aware filtering, evidence verification, and grounded language-model generation. The system is designed to provide answers that remain tied to the supplied Microsoft annual reports rather than relying on unsupported external information.
