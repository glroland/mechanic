# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG-based chatbot demo for classic C3 Corvette mechanics. It uses an OGX backend (formerly Llama Stack, OpenAI-compatible API) with Milvus vector database. The chatbot answers questions about C3 Corvettes by enriching prompts with relevant chunks from the service manual PDF.

As of OpenShift AI 3.5, Llama Stack was renamed OGX. The `LlamaStackDistribution` CRD is deprecated in favor of `OGXServer` (`ogx.io/v1beta1`), managed by the OGX Operator (`dsc.spec.components.ogx`) instead of the Llama Stack Operator. OGX integration is a Technology Preview feature in 3.5.

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
- `ai_gateway.py`: OpenAI SDK client wrapper that handles LLM chat and vector store search. Connects to OGX via `OPENAI_BASE_URL`. Uses `openai.responses.create()` for streaming chat and `openai.vector_stores.search()` for RAG retrieval.
- `constants.py`: System prompt, UI strings, session state keys.
- RAG flow: user query → vector search (`rag_search()`) → context prepended to prompt → streamed LLM response.

**`ingest/`** — Data ingestion into Milvus
- `import.py`: CLI tool (uses `click`). Converts a markdown file into chunks (split by `#`/`##`/`###` headers using `langchain_text_splitters`), then uploads each chunk as a file to OGX's vector store API via the OpenAI SDK. Deletes and recreates the `mechanic_vector_db` vector store on each run.
- `pipeline.py`: Kubeflow Pipelines (KFP) definition. Downloads the PDF directly from GitHub, uploads to OGX, and creates the vector store. Still depends on the pre-rename `llama-stack-client` package — see the TODO comment in the file before touching it. Run `python pipeline.py` to compile to `pipeline_compiled.yaml`.

**`corvetteforum-mcp/`** — MCP server (`src/app.py`)
- Exposes a single tool `search_c3_tech_support` via FastMCP/SSE.
- Uses Google Custom Search API (requires `GCP_API_KEY` and `GOOGLE_CX` env vars) to search corvetteforum.com, then fetches the top result and converts HTML to markdown via `docling`.
- Runs on port 8080 by default (`MCP_PORT` env var to override).

### Infrastructure

- **OGX** (formerly Llama Stack): OpenAI-compatible API layer sitting in front of the LLM (vLLM) and Milvus. Deployed on OpenShift AI using the OGX Operator via the `OGXServer` CRD (`deploy/ocp/ogx_deployment_using_operator.yaml`, `deploy/helm/templates/ogx_deployment.yaml`).
- **Helm chart** (`deploy/helm/`): Deploys the chatbot to OpenShift. Image pulled from `registry.home.glroland.com/mechanic/`. Install with `helm install m1 .` from `deploy/helm/`.
- **CI/CD**: `deploy/Jenkinsfile` builds the chatbot Docker image and pushes to the registry.
- **Vector DB name**: hardcoded as `mechanic_vector_db` in both `import.py` and `ai_gateway.py`.

### OGX migration notes (OpenShift AI 3.5)

Llama Stack was renamed OGX in OpenShift AI 3.5; `LlamaStackDistribution` is deprecated in favor of `OGXServer` (`ogx.io/v1beta1`). Key renames applied across `deploy/`:

| Old | New |
|---|---|
| `apiVersion: llamastack.io/v1alpha1`, `kind: LlamaStackDistribution` | `apiVersion: ogx.io/v1beta1`, `kind: OGXServer` |
| `spec.replicas` / `spec.server.containerSpec.env` / `spec.server.storage` (flat) | `spec.workload.replicas` / `spec.workload.overrides.env` / `spec.workload.storage` |
| `LLAMA_STACK_CONFIG` / `LLS_PORT` / `LLS_WORKERS` env vars | `OGX_CONFIG` / `OGX_PORT` / `OGX_WORKERS` |
| `dsc.spec.components.llamastackoperator` | `dsc.spec.components.ogx` (confirmed field name on a live 3.5 DSC; not `lls`) |
| `/opt/app-root/src/.llama` mount path | `/opt/app-root/src/.ogx` |

The standalone no-operator Deployment/Service/PVC yaml (which hardcoded the `quay.io/opendatahub/llama-stack:odh` image) was removed — the OGX Operator is now the only supported deployment path in this repo, since it resolves the correct image from `spec.distribution.name` rather than a pinned image reference.

**Verified 2026-10-21 against a live OpenShift AI 3.5 cluster** (installed the `OGXServer` CRD in a scratch namespace, applied each yaml, inspected `status`, then tore it down):

- `distribution.name: rh-dev` is accepted but the admission webhook warns it's deprecated — use `distribution.name: rh` instead (resolves to the same image: `registry.redhat.io/rhoai/odh-ogx-core-rhel9@...`).
- `spec.network.port` defaults to `8321` — the existing hardcoded port references in this repo don't need to change.
- The OGX Operator names the generated Service `<OGXServer-name>-service`, confirmed via `status.serviceURL` on a running instance — same convention as the old operator.
- `spec.workload.resources` (cpu/memory) and `spec.workload.storage` (PVC mountPath/size) are correctly placed fields — the before/after examples in Red Hat's migration doc just didn't show them.
- **Bug caught live:** `serviceAccountName`/`volumes`/`volumeMounts` live directly under `spec.workload.overrides` together with `env` — there is no separate `podOverrides` key. The CRD silently prunes unknown fields rather than rejecting them, so a `workload.podOverrides` block (which an earlier version of this repo's Helm template used) would deploy without error while silently dropping the custom service account and PVC mount. Fixed in `deploy/helm/templates/ogx_deployment.yaml`.
- **New hard requirement found:** the `rh`/`rh-dev` distribution's `StackConfig` requires a Postgres-backed KV and SQL state store at boot, independent of which `vector_io` backend (Milvus here) you use. Omitting `POSTGRES_HOST`/`POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` env vars crashes the pod with a pydantic validation error on startup. `deploy/ocp/ogx_deployment_using_operator.yaml` was missing these (the Helm chart version already had them) — added to match.

### Key Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENAI_BASE_URL` | chatbot, import.py | OGX endpoint URL (e.g. `http://localhost:8321/v1/`) |
| `OPENAI_API_KEY` | chatbot, import.py | API key (any non-empty value works with OGX) |
| `MODEL` | chatbot | LLM model name (e.g. `together/openai/gpt-oss-120b`) |
| `GCP_API_KEY` | corvetteforum-mcp | Google Custom Search API key |
| `GOOGLE_CX` | corvetteforum-mcp | Google Custom Search engine ID |

### Python Version

Python 3.12 (matches the Docker base image `ubi9/python-312`).
