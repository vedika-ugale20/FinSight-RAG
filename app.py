import streamlit as st
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f3ff 0%, #eff6ff 50%, #f0fdfa 100%);
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #2563eb, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 20px;
    color: #475569;
}

.hero-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    border: 1px solid #c4b5fd;
    margin: 20px 0;
}

.info-card {
    padding: 18px;
    border-radius: 16px;
    background: white;
    border: 1px solid #dbeafe;
    margin-bottom: 15px;
}

.answer-card {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #ecfdf5, #eff6ff);
    border: 1px solid #86efac;
}

.source-card {
    padding: 18px;
    border-radius: 16px;
    background: white;
    border-left: 5px solid #7c3aed;
    margin-bottom: 12px;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    background: #ede9fe;
    color: #6d28d9;
    font-weight: 700;
    margin-right: 6px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    font-weight: 700;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #312e81, #1e3a8a, #164e63);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():

    with open("document_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    index = faiss.read_index(
        "financial_index.faiss"
    )

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    llm_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

    return chunks, index, model, tokenizer, llm


document_chunks, index, embedding_model, tokenizer, llm = load_resources()


def retrieve_documents(query, top_k=15):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding / np.linalg.norm(
        query_embedding,
        axis=1,
        keepdims=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = document_chunks[int(idx)]

        results.append({
            "text": chunk["text"],
            "document": chunk["document"],
            "year": chunk["year"],
            "score": float(score)
        })

    return results


def generate_grounded_answer(query, sources):

    if not sources:
        return (
            "The provided documents do not contain sufficient "
            "evidence to answer this question."
        )

    context = "\n\n".join(
        f"Document: {source['document']}\n"
        f"Fiscal Year: {source['year']}\n"
        f"Evidence: {source['text'][:1800]}"
        for source in sources[:3]
    )

    prompt = f"""
You are FinSight, an enterprise financial document assistant.

Answer the user's question using ONLY the evidence provided below.

If the evidence does not contain enough information, say:
The provided documents do not contain sufficient evidence to answer this question.

Do not invent facts.
Do not use outside knowledge.
Do not confuse Microsoft Cloud revenue, segment revenue, product revenue,
net income, operating income, or total company revenue.

Question:
{query}

Evidence:
{context}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return answer


def safe_financial_answer(query):

    query_lower = query.lower()

    # -----------------------------------------
    # OUT-OF-SCOPE COMPANY CHECK
    # -----------------------------------------

    external_companies = [
        "tesla",
        "apple",
        "amazon",
        "google",
        "alphabet",
        "meta",
        "nvidia",
        "ibm",
        "oracle",
        "netflix"
    ]

    if any(
        company in query_lower
        for company in external_companies
    ):

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": (
                "The provided documents do not contain sufficient "
                "evidence to answer this question."
            ),
            "sources": []
        }

    # -----------------------------------------
    # REQUIRE MICROSOFT
    # -----------------------------------------

    microsoft_terms = [
        "microsoft",
        "msft",
        "microsoft corporation"
    ]

    if not any(
        term in query_lower
        for term in microsoft_terms
    ):

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": (
                "The provided documents contain Microsoft annual "
                "reports only. Please ask a question about Microsoft."
            ),
            "sources": []
        }

    # -----------------------------------------
    # YEAR CHECK
    # -----------------------------------------

    year_match = re.search(
        r"\b(2023|2024|2025)\b",
        query
    )

    if not year_match:

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": (
                "Please specify a fiscal year between 2023 and 2025."
            ),
            "sources": []
        }

    target_year = int(year_match.group(1))

    # -----------------------------------------
    # RETRIEVE ONLY THE REQUESTED YEAR
    # -----------------------------------------

    candidates = retrieve_documents(
        query,
        top_k=15
    )

    candidates = [
        source
        for source in candidates
        if source["year"] == target_year
    ]

    if not candidates:

        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": (
                "The provided documents do not contain sufficient "
                "evidence to answer this question."
            ),
            "sources": []
        }

    return {
        "status": "RETRIEVED",
        "answer": None,
        "sources": candidates
    }


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='font-size:32px;'>FinSight 📊</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "Enterprise Financial Intelligence"
    )

    st.markdown("### 📚 Document Collection")

    st.markdown("📄 Microsoft 2023 Annual Report")
    st.markdown("📄 Microsoft 2024 Annual Report")
    st.markdown("📄 Microsoft 2025 Annual Report")

    st.markdown("### 📅 Year Filter")

    selected_year = st.radio(
        "Select fiscal year",
        ["All Years", 2023, 2024, 2025],
        index=0
    )

    st.markdown("---")

    st.markdown("### 🔎 RAG Pipeline")
    st.markdown("📥 Document Processing")
    st.markdown("🧩 Smart Chunking")
    st.markdown("🧠 Semantic Embeddings")
    st.markdown("⚡ FAISS Vector Search")
    st.markdown("🛡️ Evidence Verification")
    st.markdown("📌 Source Attribution")

    st.markdown("---")

    st.caption("FinSight RAG")


# -----------------------------------------
# MAIN HEADER
# -----------------------------------------

st.markdown(
    '<div class="main-title">FinSight RAG 📊</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enterprise Financial Document Intelligence & Question Answering'
    '</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="hero-card">

<h3>Ask questions. Get evidence-backed answers.</h3>

<p>
FinSight searches Microsoft's annual reports using
semantic retrieval and evidence verification.
Answers are grounded in the provided financial documents.
</p>

<span class="badge">RAG</span>
<span class="badge">FAISS</span>
<span class="badge">384-D Embeddings</span>
<span class="badge">Source Attribution</span>
<span class="badge">Hallucination Control</span>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""
    <div class="info-card">
    <h4>📚 3 Reports</h4>
    <p>Microsoft annual reports from 2023–2025</p>
    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="info-card">
    <h4>🧩 246 Chunks</h4>
    <p>Metadata-aware document chunks</p>
    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown("""
    <div class="info-card">
    <h4>🧠 384-D</h4>
    <p>Sentence Transformer embeddings</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------
# QUESTION INPUT
# -----------------------------------------

st.markdown("### 💬 Ask FinSight")

query = st.text_input(
    "Your question",
    placeholder=(
        "Example: What was Microsoft's total company "
        "revenue in fiscal year 2024?"
    ),
    label_visibility="collapsed"
)


if st.button("Ask FinSight 🚀") and query:

    query_lower = query.lower()

    # -----------------------------------------
    # COMPANY SCOPE CHECK
    # -----------------------------------------

    external_companies = [
        "tesla",
        "apple",
        "amazon",
        "google",
        "alphabet",
        "meta",
        "nvidia",
        "ibm",
        "oracle",
        "netflix"
    ]

    if any(
        company in query_lower
        for company in external_companies
    ):

        answer = (
            "The provided documents do not contain sufficient "
            "evidence to answer this question."
        )

        status = "INSUFFICIENT_EVIDENCE"
        sources = []

    elif "microsoft" not in query_lower:

        answer = (
            "The provided documents contain Microsoft annual "
            "reports only. Please ask a question about Microsoft."
        )

        status = "INSUFFICIENT_EVIDENCE"
        sources = []

    else:

        # -----------------------------------------
        # YEAR FILTER
        # -----------------------------------------

        year_match = re.search(
            r"\b(2023|2024|2025)\b",
            query
        )

        if selected_year != "All Years":

            target_year = selected_year

        elif year_match:

            target_year = int(
                year_match.group(1)
            )

        else:

            target_year = None


        # -----------------------------------------
        # RETRIEVAL
        # -----------------------------------------

        with st.spinner(
            "🔍 Searching financial documents..."
        ):

            sources = retrieve_documents(
                query,
                top_k=15
            )


        # -----------------------------------------
        # APPLY YEAR FILTER
        # -----------------------------------------

        if target_year is not None:

            sources = [
                source
                for source in sources
                if source["year"] == target_year
            ]


        # -----------------------------------------
        # CHECK EVIDENCE
        # -----------------------------------------

        if not sources:

            answer = (
                "The selected fiscal year does not contain "
                "sufficient evidence to answer this question."
            )

            status = "INSUFFICIENT_EVIDENCE"

        else:

            # -----------------------------------------
            # GENERATE GROUNDED ANSWER
            # -----------------------------------------

            with st.spinner(
                "🤖 Generating grounded answer..."
            ):

                answer = generate_grounded_answer(
                    query,
                    sources
                )


            if (
                not answer
                or
                "do not contain sufficient evidence"
                in answer.lower()
            ):

                status = "INSUFFICIENT_EVIDENCE"

            else:

                status = "GROUNDED"


    # -----------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------

    st.markdown("### ✨ Answer")


    if status == "GROUNDED":

        st.markdown(
            f"""
            <div class="answer-card">

            <h3>✅ Grounded Answer</h3>

            <p style="font-size:22px;">
            {answer}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "⚠️ " + answer
        )


    # -----------------------------------------
    # DISPLAY SOURCES
    # -----------------------------------------

    st.markdown("### 📌 Supporting Sources")


    if sources:

        for i, result in enumerate(
            sources[:3],
            1
        ):

            st.markdown(
                f"""
                <div class="source-card">

                <b>📄 Source {i}</b><br>

                <b>Document:</b>
                {result['document']}<br>

                <b>Fiscal Year:</b>
                FY{result['year']}

                </div>
                """,
                unsafe_allow_html=True
            )


            with st.expander(
                f"🔎 View evidence from Source {i}"
            ):

                st.write(
                    result["text"]
                )

    else:

        st.info(
            "No supporting evidence was found in the "
            "provided documents."
        )
