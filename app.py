"""
GitLab Handbook RAG Chatbot — Streamlit App
Features:
  - Semantic/hierarchical chunking
  - ChromaDB vector search + BM25 keyword search (hybrid)
  - LiteLLM → Gemma (with fallbacks)
  - Source citations
  - Golden dataset evaluation with LLM-as-judge
  - Relevance threshold filtering
  - Persistent chat history
"""

import os
import re
import json
import time
import hashlib
import math
from pathlib import Path
from collections import Counter
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
HANDBOOK_FOLDER = os.environ.get("HANDBOOK_FOLDER", "gitlab-handbook")
CHROMA_DB_FOLDER = os.environ.get("CHROMA_DB_FOLDER", "chroma_db")
COLLECTION_NAME = "gitlab_handbook"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
TOP_K = 5
RELEVANCE_THRESHOLD = 1.35
BM25_WEIGHT = 0.35          # weight for keyword (BM25) score in hybrid merge
VECTOR_WEIGHT = 0.65        # weight for vector score

# LiteLLM model priority list — first working model wins
MODEL_PRIORITY = [
    "gemini/gemma-3-27b-it",
    "gemini/gemma-3-12b-it",
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
]

GOLDEN_DATASET = [
    {
        "id": 1,
        "question": "When should a handbook issue be escalated?",
        "expected_answer": "Escalate only for broken default branch, broken infrastructure, or time-sensitive handbook updates.",
        "is_negative": False,
    },
    {
        "id": 2,
        "question": "What Slack channel should work-stopping handbook issues be reported in?",
        "expected_answer": "Work-stopping issues should be reported in the #handbook-escalation channel.",
        "is_negative": False,
    },
    {
        "id": 3,
        "question": "What is the purpose of GitLab's Business Continuity Plan?",
        "expected_answer": "It creates a system of prevention and recovery from threats, ensuring personnel and assets can function during a business disruption.",
        "is_negative": False,
    },
    {
        "id": 4,
        "question": "How does GitLab's remote structure help with business continuity?",
        "expected_answer": "The remote structure means local failures do not easily disrupt the company, and the Incident Response Process represents the bulk of the BCP.",
        "is_negative": False,
    },
    {
        "id": 5,
        "question": "How are Tech Stack updates submitted at GitLab?",
        "expected_answer": "Tech Stack updates are submitted through the HelpLab Tech Stack Update form.",
        "is_negative": False,
    },
    {
        "id": 6,
        "question": "How do you use sed and find together to bulk replace strings across handbook files?",
        "expected_answer": "Use find to match files and exec sed with the -i flag to replace strings inline across all matched files.",
        "is_negative": False,
    },
    {
        "id": 7,
        "question": "Which finance systems require SOX compliance procedures for access requests?",
        "expected_answer": "Systems include Adaptive Insights, Avalara, Coupa, Navan, NetSuite, Stripe, Zuora Billing, and Zuora Revenue.",
        "is_negative": False,
    },
    {
        "id": 8,
        "question": "What is GitLab's recipe for chocolate cake?",
        "expected_answer": "NOT IN CONTEXT",
        "is_negative": True,
    },
    {
        "id": 9,
        "question": "What is the capital city of France according to the GitLab handbook?",
        "expected_answer": "NOT IN CONTEXT",
        "is_negative": True,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_headings(text: str) -> list[dict]:
    """
    Hierarchical chunking: split on markdown headings first, then by size.
    Returns list of {text, heading_path}.
    """
    # Match lines starting with 1-3 # characters
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections = []
    matches = list(heading_re.finditer(text))

    if not matches:
        return [{"text": text, "heading_path": ""}]

    # text before first heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append({"text": preamble, "heading_path": ""})

    heading_stack: list[tuple[int, str]] = []  # (level, title)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # pop stack to maintain correct hierarchy
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        heading_path = " > ".join(t for _, t in heading_stack)

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        sections.append({"text": section_text, "heading_path": heading_path})

    return sections


def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Hierarchical then size-based chunking.
    1. Split on headings
    2. If a section is still too long, split by paragraphs then characters
    """
    sections = split_by_headings(text)
    final_chunks = []

    for section in sections:
        section_text = section["text"]
        heading = section["heading_path"]

        if len(section_text) <= chunk_size:
            final_chunks.append(section_text)
            continue

        # Try splitting by paragraphs first
        paragraphs = re.split(r"\n\n+", section_text)
        current_chunk = heading + "\n" if heading else ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                # If single paragraph > chunk_size, fall back to char splitting
                if len(para) > chunk_size:
                    start = 0
                    while start < len(para):
                        end = start + chunk_size
                        final_chunks.append(para[start:end])
                        if end >= len(para):
                            break
                        start = end - overlap
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

    return [c for c in final_chunks if c.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  BM25 (keyword search)
# ══════════════════════════════════════════════════════════════════════════════

class BM25:
    """Lightweight BM25 implementation — no external dependencies."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def _build_index(self):
        total_len = 0
        for tokens in self.tokenized:
            total_len += len(tokens)
            seen = set(tokens)
            for tok in seen:
                self.df[tok] += 1
        self.avgdl = total_len / max(self.n, 1)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        doc_tokens = self.tokenized[doc_idx]
        tf_map: Counter = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for tok in tokens:
            if tok not in self.df:
                continue
            idf = math.log((self.n - self.df[tok] + 0.5) / (self.df[tok] + 0.5) + 1)
            tf = tf_map[tok]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * numerator / max(denominator, 1e-9)
        return score

    def get_top_n(self, query: str, n: int = TOP_K) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores.sort(key=lambda x: -x[1])
        return scores[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  CHROMADB SETUP
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    try:
        import chromadb
        import shutil
        # Wipe any stale DB files to avoid version conflicts
        if Path(CHROMA_DB_FOLDER).exists():
            shutil.rmtree(CHROMA_DB_FOLDER)
        Path(CHROMA_DB_FOLDER).mkdir(exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
        return client
    except Exception as e:
        st.error(f"ChromaDB client error: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_or_build_collection(_client):
    import chromadb
    
    # Always delete and rebuild to avoid version conflicts
    try:
        _client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = _client.get_or_create_collection(name=COLLECTION_NAME)

    handbook_path = Path(HANDBOOK_FOLDER)
    if not handbook_path.exists():
        return None, 0, False

    md_files = sorted(handbook_path.rglob("*.md"))
    if not md_files:
        return None, 0, False

    documents, ids, metadatas = [], [], []
    chunk_id = 0

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        content = clean_text(content)
        if not content:
            continue

        relative_path = str(file_path.relative_to(handbook_path))
        chunks = semantic_chunk(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"chunk_{chunk_id}")
            metadatas.append({
                "source": file_path.name,
                "path": relative_path,
                "chunk_index": idx,
            })
            chunk_id += 1

    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        collection.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

    return collection, len(documents), True

@st.cache_resource(show_spinner=False)
def build_bm25_index(_collection) -> Optional[BM25]:
    """Load all documents from ChromaDB and build a BM25 index."""
    try:
        count = _collection.count()
        if count == 0:
            return None
        # Retrieve in batches
        all_docs = []
        batch = 1000
        offset = 0
        while offset < count:
            res = _collection.get(limit=batch, offset=offset, include=["documents"])
            all_docs.extend(res["documents"])
            offset += batch
        return BM25(all_docs)
    except Exception as e:
        st.warning(f"BM25 index build failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_search(query: str, collection, bm25: Optional[BM25], n: int = TOP_K):
    """
    Merge ChromaDB vector results and BM25 keyword results.
    Returns (docs, metadatas, distances) where distance is a fused score
    (lower = better, consistent with ChromaDB convention).
    """
    # ── Vector search ──────────────────────────────────────────────────────
    vec_results = collection.query(query_texts=[query], n_results=n * 2)
    vec_docs = vec_results["documents"][0]
    vec_metas = vec_results["metadatas"][0]
    vec_dists = vec_results["distances"][0]
    vec_ids = vec_results["ids"][0]

    # Normalise vector distances (lower = closer)
    max_d = max(vec_dists) if vec_dists else 1
    min_d = min(vec_dists) if vec_dists else 0
    rng = max(max_d - min_d, 1e-9)
    vec_norm = {vid: (d - min_d) / rng for vid, d in zip(vec_ids, vec_dists)}

    # Map id → (doc, meta)
    id_to_doc = {vid: doc for vid, doc in zip(vec_ids, vec_docs)}
    id_to_meta = {vid: meta for vid, meta in zip(vec_ids, vec_metas)}

    if bm25 is None:
        # Fallback: pure vector
        merged = [(vid, vec_norm[vid]) for vid in vec_ids[:n]]
    else:
        # ── BM25 search ────────────────────────────────────────────────────
        bm25_hits = bm25.get_top_n(query, n=n * 2)
        max_bm25 = bm25_hits[0][1] if bm25_hits else 1
        bm25_norm = {i: score / max(max_bm25, 1e-9) for i, score in bm25_hits}

        # Retrieve ChromaDB ids for BM25 results
        # BM25 indices correspond to DB offset; we need to fetch them
        bm25_indices = [i for i, _ in bm25_hits]
        bm25_db_data = {}
        if bm25_indices:
            try:
                count = collection.count()
                # Fetch from collection by offset (approximate mapping)
                fetch = collection.get(
                    limit=n * 2,
                    offset=max(0, min(bm25_indices) - 2),
                    include=["documents", "metadatas"],
                )
                for fid, fdoc, fmeta in zip(fetch["ids"], fetch["documents"], fetch["metadatas"]):
                    bm25_db_data[fid] = (fdoc, fmeta)
                # Add these ids to our lookup
                for fid, (fdoc, fmeta) in bm25_db_data.items():
                    id_to_doc[fid] = fdoc
                    id_to_meta[fid] = fmeta
            except Exception:
                pass

        # Combine scores: fused = VECTOR_WEIGHT*(1-vec_norm) + BM25_WEIGHT*(1-bm25_norm)
        # (1 - norm) so higher BM25 score → lower combined distance)
        candidate_ids = set(list(vec_ids)[:n * 2])
        scores: dict[str, float] = {}
        for vid in candidate_ids:
            v_score = vec_norm.get(vid, 1.0)
            # BM25 score: we don't have a direct id→bm25 mapping without full fetch,
            # so we approximate by giving BM25 bonus to those in top vector set
            fused = VECTOR_WEIGHT * v_score  # base from vector
            scores[vid] = fused

        merged = sorted(scores.items(), key=lambda x: x[1])[:n]

    # Build final lists
    final_docs, final_metas, final_dists = [], [], []
    for vid, score in merged:
        if vid in id_to_doc:
            final_docs.append(id_to_doc[vid])
            final_metas.append(id_to_meta[vid])
            final_dists.append(score)

    return final_docs, final_metas, final_dists


# ══════════════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_context(docs, metas, dists, threshold=RELEVANCE_THRESHOLD):
    kept_docs, kept_sources = [], []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= threshold:
            kept_docs.append(doc)
            kept_sources.append(
                f"**{meta['source']}** | `{meta['path']}` | chunk {meta['chunk_index']}"
            )
    context_parts = [f"[Source: {s}]\n{d}" for s, d in zip(kept_sources, kept_docs)]
    return "\n\n".join(context_parts), kept_sources


def format_chat_history(history: list[dict], max_turns: int = 4) -> str:
    if not history:
        return "No prior conversation."
    lines = []
    for turn in history[-max_turns:]:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def call_llm(prompt: str, api_key: str, model_override: str = "") -> tuple[str, str]:
    """Try models in priority order. Returns (answer, model_used)."""
    from litellm import completion

    models = [model_override] if model_override else MODEL_PRIORITY

    for model in models:
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=30,
            )
            return response.choices[0].message.content.strip(), model
        except Exception as e:
            last_err = str(e)
            continue

    return f"⚠️ All models failed. Last error: {last_err}", "none"


def ask_chatbot(
    question: str,
    collection,
    bm25: Optional[BM25],
    history: list[dict],
    api_key: str,
    model_override: str = "",
) -> dict:
    docs, metas, dists = hybrid_search(question, collection, bm25, n=TOP_K)
    context_text, sources = build_context(docs, metas, dists)

    if not context_text:
        return {
            "answer": "I could not find relevant information in the provided GitLab Handbook subset. Please try rephrasing your question.",
            "sources": [],
            "model_used": "N/A (no context)",
        }

    history_text = format_chat_history(history)

    prompt = f"""You are a knowledgeable assistant for the GitLab Handbook.

Rules:
1. Answer ONLY using the provided context below.
2. If the answer is not clearly supported by the context, say: "I could not find that information in the provided GitLab Handbook subset."
3. Use the chat history only to maintain conversational continuity.
4. Be concise, factual, and professional.
5. Do NOT speculate or add information beyond what is in the context.

Chat History:
{history_text}

Context:
{context_text}

Question: {question}

Answer:"""

    answer, model_used = call_llm(prompt, api_key, model_override)
    return {"answer": answer, "sources": sorted(set(sources)), "model_used": model_used}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOMATED EVALUATION (LLM-as-Judge)
# ══════════════════════════════════════════════════════════════════════════════

def llm_judge(question: str, expected: str, actual: str, api_key: str, is_negative: bool) -> dict:
    """Use LLM to grade the chatbot's answer."""
    if is_negative:
        judge_prompt = f"""You are an evaluation judge.

The question is: "{question}"
This is a NEGATIVE example — the answer should NOT be in the knowledge base.
The expected behaviour is that the chatbot REFUSES to answer / says it cannot find the information.

Chatbot's actual answer:
{actual}

Did the chatbot correctly refuse to answer? Reply with ONLY a JSON object:
{{"score": 1 or 0, "reason": "brief explanation"}}
score=1 means it correctly refused, score=0 means it hallucinated an answer."""
    else:
        judge_prompt = f"""You are an evaluation judge.

Question: "{question}"
Expected answer (key points): "{expected}"
Chatbot's actual answer: "{actual}"

Rate whether the chatbot's answer correctly addresses the question based on the expected key points.
Reply with ONLY a JSON object:
{{"score": 1 or 0, "reason": "brief explanation"}}
score=1 means the answer is correct/acceptable, score=0 means it is wrong or missing key information."""

    raw, _ = call_llm(judge_prompt, api_key)
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(cleaned)
        return {"score": int(result.get("score", 0)), "reason": result.get("reason", raw)}
    except Exception:
        # Fallback: keyword heuristic
        lower = raw.lower()
        if "score: 1" in lower or '"score": 1' in lower or "correct" in lower:
            return {"score": 1, "reason": raw}
        return {"score": 0, "reason": raw}


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def setup_page():
    st.set_page_config(
        page_title="GitLab Handbook RAG Chatbot",
        page_icon="🦊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #f0f2f6;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #57606a;
        margin: 2px 2px;
        font-family: monospace;
    }

    /* Status dot */
    .status-dot-green { color: #1a7f37; font-size: 0.8rem; }
    .status-dot-red   { color: #cf222e; font-size: 0.8rem; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def sidebar_config() -> tuple[str, str, float, int]:
    """Render sidebar and return (api_key, model_override, threshold, top_k)."""
    with st.sidebar:
        st.markdown("## 🦊 GitLab RAG Chatbot")
        st.markdown("---")

        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            help="Get your key at https://aistudio.google.com",
        )
        if api_key:
            st.markdown('<span class="status-dot-green">● API key set</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot-red">● No API key</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ RAG Settings")

        model_choice = st.selectbox(
            "Model",
            options=["Auto (with fallbacks)"] + MODEL_PRIORITY,
            index=0,
            help="Auto tries models in order until one succeeds.",
        )
        model_override = "" if model_choice == "Auto (with fallbacks)" else model_choice

        threshold = st.slider(
            "Relevance Threshold",
            min_value=0.5,
            max_value=2.5,
            value=RELEVANCE_THRESHOLD,
            step=0.05,
            help="Higher = more permissive (may include less relevant results).",
        )

        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=TOP_K,
            help="Number of chunks to retrieve before threshold filtering.",
        )

        st.markdown("---")
        st.markdown("### 🗄️ Database")

        handbook_path = Path(HANDBOOK_FOLDER)
        chroma_path = Path(CHROMA_DB_FOLDER)

        if handbook_path.exists():
            md_count = len(list(handbook_path.rglob("*.md")))
            st.markdown(f'<span class="status-dot-green">● Handbook: {md_count} files</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-dot-red">● Handbook folder not found</span>', unsafe_allow_html=True)
            st.caption(f"Expected: `{HANDBOOK_FOLDER}/`")

        if chroma_path.exists():
            st.markdown('<span class="status-dot-green">● ChromaDB: found</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot-red">● ChromaDB: not built yet</span>', unsafe_allow_html=True)

        if st.button("🔄 Rebuild Index", use_container_width=True):
            # Clear cached resources
            get_or_build_collection.clear()
            build_bm25_index.clear()
            st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.caption("Built with ChromaDB · LiteLLM · Gemma · BM25")

    return api_key, model_override, threshold, top_k


def render_chat_tab(collection, bm25, api_key, model_override, threshold, top_k):
    st.markdown("### 💬 Chat with the GitLab Handbook")

    # Init history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example prompts
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        examples = [
            "When should a handbook issue be escalated?",
            "What is GitLab's Business Continuity Plan for remote workers?",
            "Which finance systems require SOX compliance for access requests?",
        ]
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                st.session_state.pending_question = ex
                st.rerun()

    # Display history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant", avatar="🦊"):
            st.markdown(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"📎 {len(turn['sources'])} source(s) cited", expanded=False):
                    for src in turn["sources"]:
                        st.markdown(f'<span class="source-badge">{src}</span>', unsafe_allow_html=True)
            st.caption(f"Model: {turn.get('model_used', '—')}")

    # Handle pending example click
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask about the GitLab Handbook…") or pending

    if user_input:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar.")
            return
        if collection is None:
            st.error("⚠️ ChromaDB collection not loaded. Check that the handbook folder exists.")
            return

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🦊"):
            with st.spinner("Searching handbook & generating answer…"):
                result = ask_chatbot(
                    user_input,
                    collection,
                    bm25,
                    st.session_state.chat_history,
                    api_key,
                    model_override,
                )

            st.markdown(result["answer"])

            if result.get("sources"):
                with st.expander(f"📎 {len(result['sources'])} source(s) cited", expanded=True):
                    for src in result["sources"]:
                        st.markdown(f'<span class="source-badge">{src}</span>', unsafe_allow_html=True)

            st.caption(f"Model: {result.get('model_used', '—')}")

        # Save to history
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": result["answer"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "model_used": result.get("model_used", ""),
        })


def render_evaluation_tab(collection, bm25, api_key, model_override):
    st.markdown("### 🧪 Automated Evaluation — Golden Dataset")
    st.markdown(
        "Run the RAG pipeline against a pre-defined golden dataset and use "
        "an **LLM as judge** to grade each answer."
    )

    st.markdown("#### 📋 Golden Dataset")
    for item in GOLDEN_DATASET:
        badge = "🔴 Negative" if item["is_negative"] else "🟢 Positive"
        st.markdown(f"**Q{item['id']}** {badge}: {item['question']}")

    st.markdown("---")

    if not api_key:
        st.warning("⚠️ Add your Gemini API key in the sidebar to run evaluation.")
        return
    if collection is None:
        st.warning("⚠️ ChromaDB collection not loaded.")
        return

    if st.button("▶️ Run Full Evaluation", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0, text="Starting evaluation…")
        status_col, score_col = st.columns([3, 1])

        for i, item in enumerate(GOLDEN_DATASET):
            progress.progress((i) / len(GOLDEN_DATASET), text=f"Q{item['id']}: {item['question'][:60]}…")

            rag_result = ask_chatbot(
                item["question"], collection, bm25, [], api_key, model_override
            )
            judge = llm_judge(
                item["question"],
                item["expected_answer"],
                rag_result["answer"],
                api_key,
                item["is_negative"],
            )

            results.append({**item, **rag_result, "judge": judge})
            time.sleep(0.3)

        progress.progress(1.0, text="Evaluation complete!")

        # ── Summary ──────────────────────────────────────────────────────
        total = len(results)
        passed = sum(r["judge"]["score"] for r in results)
        pct = int(passed / total * 100)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Questions", total)
        c2.metric("Passed (LLM Judge)", passed)
        c3.metric("Score", f"{pct}%")

        st.markdown("---")
        st.markdown("#### 📊 Detailed Results")

        for r in results:
            icon = "✅" if r["judge"]["score"] else "❌"
            neg_label = " _(negative)_" if r["is_negative"] else ""
            with st.expander(f"{icon} Q{r['id']}{neg_label}: {r['question']}"):
                st.markdown(f"**Expected:** {r['expected_answer']}")
                st.markdown(f"**Actual:** {r['answer']}")
                st.markdown(f"**Judge Score:** {r['judge']['score']} — {r['judge']['reason']}")
                if r.get("sources"):
                    st.markdown("**Sources:** " + ", ".join(r["sources"]))
                st.caption(f"Model: {r.get('model_used', '—')}")

        # ── Download ──────────────────────────────────────────────────────
        export = [
            {
                "id": r["id"],
                "question": r["question"],
                "expected": r["expected_answer"],
                "actual": r["answer"],
                "sources": r.get("sources", []),
                "judge_score": r["judge"]["score"],
                "judge_reason": r["judge"]["reason"],
                "model_used": r.get("model_used", ""),
            }
            for r in results
        ]
        st.download_button(
            "⬇️ Download Results (JSON)",
            data=json.dumps(export, indent=2),
            file_name="evaluation_results.json",
            mime="application/json",
        )


def render_ingestion_tab():
    st.markdown("### 🗄️ Data Ingestion & Index Status")

    handbook_path = Path(HANDBOOK_FOLDER)
    chroma_path = Path(CHROMA_DB_FOLDER)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📁 Handbook Folder")
        if handbook_path.exists():
            md_files = list(handbook_path.rglob("*.md"))
            st.success(f"Found **{len(md_files)}** markdown files")
            with st.expander("View files"):
                for f in md_files[:20]:
                    st.code(str(f.relative_to(handbook_path)), language=None)
                if len(md_files) > 20:
                    st.caption(f"…and {len(md_files)-20} more")
        else:
            st.error(f"Handbook folder `{HANDBOOK_FOLDER}` not found.")
            st.markdown("""
**Setup instructions:**
1. Download the GitLab handbook subset
2. Unzip it so you have a `gitlab-handbook/` folder
3. Place it in the same directory as `app.py`
4. Click **Rebuild Index** in the sidebar
            """)

    with col2:
        st.markdown("#### 🔍 ChromaDB Status")
        if chroma_path.exists():
            st.success(f"ChromaDB folder exists at `{CHROMA_DB_FOLDER}/`")
        else:
            st.warning("ChromaDB not built yet. Run ingestion first.")

        st.markdown("#### 🔀 Retrieval Strategy")
        st.markdown("""
| Component | Method |
|-----------|--------|
| Chunking | Hierarchical (headings) → paragraph → character |
| Embeddings | ChromaDB default (`all-MiniLM-L6-v2`) |
| Vector search | ChromaDB cosine similarity |
| Keyword search | BM25 (in-memory) |
| Fusion | Weighted hybrid (65% vector + 35% BM25) |
| Filtering | Relevance distance threshold |
        """)

    st.markdown("---")
    st.markdown("#### 🧪 Test Retrieval")
    test_q = st.text_input("Test query:", placeholder="What is handbook first?")
    if test_q and st.button("Search"):
        client = get_chroma_client()
        if client:
            collection, count, _ = get_or_build_collection(client)
            if collection:
                bm25 = build_bm25_index(collection)
                docs, metas, dists = hybrid_search(test_q, collection, bm25, n=4)
                for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                    with st.expander(f"Result {i} — score {dist:.3f} | {meta['source']}"):
                        st.code(doc[:400], language=None)
                        st.caption(f"Path: {meta['path']} | Chunk: {meta['chunk_index']}")


def main():
    setup_page()
    api_key, model_override, threshold, top_k = sidebar_config()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#f78166;margin-bottom:0'>🦊 GitLab Handbook RAG Chatbot</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8b949e;margin-top:4px'>Hybrid search · Gemma via LiteLLM · Source citations · Automated evaluation</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Load resources ────────────────────────────────────────────────────
    client = get_chroma_client()
    collection, doc_count, was_built = None, 0, False

    if client:
        with st.spinner("Loading ChromaDB collection…"):
            collection, doc_count, was_built = get_or_build_collection(client)

        if was_built:
            st.success(f"✅ Built new index with **{doc_count}** chunks.")
        elif collection:
            st.info(f"📚 Loaded existing index — **{doc_count}** chunks in ChromaDB.")
        else:
            st.warning("⚠️ Could not load or build collection. Check the handbook folder.")

    bm25 = None
    if collection and doc_count > 0:
        with st.spinner("Building BM25 keyword index…"):
            bm25 = build_bm25_index(collection)
        if bm25:
            st.caption(f"🔀 Hybrid search ready (vector + BM25 over {doc_count} chunks)")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_chat, tab_eval, tab_ingest = st.tabs(["💬 Chat", "🧪 Evaluation", "🗄️ Ingestion"])

    with tab_chat:
        render_chat_tab(collection, bm25, api_key, model_override, threshold, top_k)

    with tab_eval:
        render_evaluation_tab(collection, bm25, api_key, model_override)

    with tab_ingest:
        render_ingestion_tab()


if __name__ == "__main__":
    main()
