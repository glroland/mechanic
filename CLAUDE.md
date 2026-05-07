# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG-based chatbot demo for classic C3 Corvette mechanics. It uses a LLama Stack backend (OpenAI-compatible API) with Milvus vector database. The chatbot answers questions about C3 Corvettes by enriching prompts with relevant chunks from the service manual PDF.

## Commands

All common workflows go through the root `Makefile`. Configure the variables at the top of the Makefile (or export them) before running:

```bash
# Install all Python dependencies (platform-aware: mac vs linux)
make install

# Convert the service manual PDF and ingest into Milvus
make ingest-data

# Run the chatbot locally (requires OPENAI_BASE_URL, OPENAI_API_KEY, MODEL env vars)
make run-chatbot

# Run the Corvette Forum MCP server
make run-corvetteforummcp

# Compile the Kubeflow Pipeline YAML (output: ingest/src/pipeline_compiled.yaml)
make compile-pipeline

# Clean build artifacts
make clean
```

**Direct CLI ingestion** (bypass make):
```bash
cd ingest/src && python import.py <openai_baseurl> <embedding_model> <vdb_provider> <input_file>
```

## Architecture

### Components

**`chatbot/`** — Streamlit web UI (`src/app.py`)
- `ai_gateway.py`: OpenAI SDK client wrapper that handles LLM chat and vector store search. Connects to LLama Stack via `OPENAI_BASE_URL`. Uses `openai.responses.create()` for streaming chat and `openai.vector_stores.search()` for RAG retrieval.
- `constants.py`: System prompt, UI strings, session state keys.
- RAG flow: user query → vector search (`rag_search()`) → context prepended to prompt → streamed LLM response.

**`ingest/`** — Data ingestion into Milvus
- `import.py`: CLI tool (uses `click`). Converts a markdown file into chunks (split by `#`/`##`/`###` headers using `langchain_text_splitters`), then uploads each chunk as a file to LLama Stack's vector store API. Deletes and recreates the `mechanic_vector_db` vector store on each run.
- `pipeline.py`: Kubeflow Pipelines (KFP) definition. Downloads the PDF directly from GitHub, uploads to LLama Stack, and creates the vector store. Run `python pipeline.py` to compile to `pipeline_compiled.yaml`.

**`corvetteforum-mcp/`** — MCP server (`src/app.py`)
- Exposes a single tool `search_c3_tech_support` via FastMCP/SSE.
- Uses Google Custom Search API (requires `GCP_API_KEY` and `GOOGLE_CX` env vars) to search corvetteforum.com, then fetches the top result and converts HTML to markdown via `docling`.
- Runs on port 8080 by default (`MCP_PORT` env var to override).

### Infrastructure

- **LLama Stack**: OpenAI-compatible API layer sitting in front of the LLM (vLLM) and Milvus. Deployed on OpenShift AI using the LLama Stack operator (`deploy/ocp/lls_deployment_using_operator.yaml`).
- **Helm chart** (`deploy/helm/`): Deploys the chatbot to OpenShift. Image pulled from `registry.home.glroland.com/mechanic/`. Install with `helm install m1 .` from `deploy/helm/`.
- **CI/CD**: `deploy/Jenkinsfile` builds the chatbot Docker image and pushes to the registry.
- **Vector DB name**: hardcoded as `mechanic_vector_db` in both `import.py` and `ai_gateway.py`.

### Key Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENAI_BASE_URL` | chatbot, import.py | LLama Stack endpoint URL (e.g. `http://localhost:8321/v1/`) |
| `OPENAI_API_KEY` | chatbot, import.py | API key (any non-empty value works with LLama Stack) |
| `MODEL` | chatbot | LLM model name (e.g. `together/openai/gpt-oss-120b`) |
| `GCP_API_KEY` | corvetteforum-mcp | Google Custom Search API key |
| `GOOGLE_CX` | corvetteforum-mcp | Google Custom Search engine ID |

### Python Version

Python 3.12 (matches the Docker base image `ubi9/python-312`).
