# 🦊 GitLab Handbook RAG Chatbot — Streamlit

A full RAG pipeline over the GitLab Handbook subset, deployed as a Streamlit app.

## Features

| Feature | Implementation |
|---------|---------------|
| Chunking | Hierarchical (headings → paragraphs → characters) |
| Embeddings | ChromaDB default (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent local) |
| Keyword Search | BM25 (in-memory, no extra deps) |
| Hybrid Search | 65% vector + 35% BM25 weighted fusion |
| LLM | Gemma 3 27B via LiteLLM + auto-fallbacks |
| Relevance Filter | Configurable distance threshold |
| Chat History | Maintained across turns (last 4 turns in context) |
| Source Citations | Displayed per answer |
| Evaluation | LLM-as-Judge against 9-question golden dataset |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your handbook
```
your-project/
├── app.py
├── requirements.txt
├── gitlab-handbook/        ← unzip your handbook here
│   ├── communication.md
│   ├── values.md
│   └── ...
└── chroma_db/              ← auto-created on first run
```

### 3. Set your API key
```bash
export GEMINI_API_KEY=AIza...
# OR create a .env file:
echo "GEMINI_API_KEY=AIza..." > .env
```

### 4. Run
```bash
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push this repo (including `chroma_db/` folder) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Add `GEMINI_API_KEY` in **Secrets** (Settings → Secrets)

> **Important:** The `chroma_db/` folder must be committed to the repo OR rebuilt at startup. Since rebuilding requires the handbook files, it's easiest to commit the pre-built `chroma_db/` folder.

## Model Fallback Chain

The app tries these models in order until one succeeds:
1. `gemini/gemma-3-27b-it`
2. `gemini/gemma-3-12b-it`
3. `gemini/gemini-2.0-flash`
4. `gemini/gemini-1.5-flash`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required — your Google AI Studio key |
| `HANDBOOK_FOLDER` | `gitlab-handbook` | Path to markdown files |
| `CHROMA_DB_FOLDER` | `chroma_db` | Path for ChromaDB persistence |
"# Genai-Gitlab-project" 
# Genai-Gitlab-project
# Genai-Gitlab-project
# Genai-Gitlab-project
# Genai-Gitlab-project-lbs
