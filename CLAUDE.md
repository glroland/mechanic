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
- `ai_gateway.py`: OpenAI SDK client wrapper that handles LLM chat and vector store search. Connects to OGX via `OPENAI_BASE_URL`. Uses `openai.responses.create()` for streaming chat and `openai.vector_stores.search()` for RAG retrieval. `process_user_chat()`/`rag_search()` are each wrapped in `@mlflow.trace(...)`.
- `tracing.py`: `configure_tracing()` — mirrors the `distribution-center` project's pattern. Calls `mlflow.openai.autolog()` unconditionally (safe/idempotent), and disables tracing outright (`mlflow.tracing.disable()`) when `MLFLOW_TRACKING_URI` isn't set, so nothing writes to a local `./mlruns` dir in dev. Called once at the top of `app.py` on every Streamlit rerun. Everything else (`MLFLOW_WORKSPACE`, `MLFLOW_TRACKING_AUTH`, `MLFLOW_EXPERIMENT_NAME`) is read natively by mlflow from the environment.
- `constants.py`: System prompt, UI strings, session state keys.
- RAG flow: user query → vector search (`rag_search()`) → context prepended to prompt → streamed LLM response.

**`ingest/`** — Data ingestion into Milvus
- `import.py`: CLI tool (uses `click`). Converts a markdown file into chunks (split by `#`/`##`/`###` headers using `langchain_text_splitters`), then uploads each chunk as a file to OGX's vector store API via the OpenAI SDK. Deletes and recreates the `mechanic_vector_db` vector store on each run. Before chunking, runs `redact_noisy_sections()` over the whole file to find and replace OCR/extraction-noise regions (see "RAG content quality" below); the per-chunk `noise_ratio()` check after chunking is just a backstop for a chunk that's still mostly noise post-redaction.
- `pipeline.py`: Kubeflow Pipelines (KFP) definition. Downloads a pre-converted markdown file (not the raw PDF — see "RAG content quality" below for why) from GitHub, uploads it to OGX, and creates the vector store. Runs the same noise-redaction pass as `import.py` (duplicated inline in the component function, not imported — KFP lightweight components are self-contained and run in their own isolated container; keep the two copies in sync by hand) on the downloaded content before upload. Uses `ogx-client` (the successor to `llama-stack-client`) via `OgxClient` — see the comment on the component decorator for the call-shape differences, confirmed by three real failed pipeline runs on 2026-10-21: `client.vector_stores_files` instead of `client.vector_stores.files`, `_request_timeout=` instead of `timeout=`, a 2-tuple `(filename, bytes)` for file uploads (no content-type, no `BytesIO`), every response read via the `*_without_preload_content()` + `.json()` pattern instead of the typed `create()`/`list()`/`retrieve()` methods (because `ogx-client==1.5.0`'s bundled response models are stricter than what this cluster's actual OGX server version returns), and an explicit `.read()` on each of those responses before `.json()` (ogx-client always sends requests with `stream=True`, so a `_without_preload_content()` response is unread until you call `.read()` yourself — the typed methods do this internally before deserializing, but skip it entirely in this raw path). Run `python pipeline.py` to compile to `pipeline_compiled.yaml`.

### RAG content quality (OCR/extraction noise)

A user report of garbled MLflow trace inputs (trace `tr-e88efb22a600a006273afff02d0aa8e1`) traced back to genuine data corruption, confirmed 2026-10-21: `chatbot/src/assets/c3_repair.pdf` (an old Word doc run through "Acrobat PDFWriter 4.05" in 2002, per its own embedded metadata) has several tables whose embedded font glyph encoding is corrupted in the source file itself — confirmed by rendering the affected pages to an image (not just reading the text layer) and seeing the same garbage glyphs in the rendering. No extraction method (pypdf, docling/OCR) can recover these; the only fix is detecting and redacting them.

Two separate things were done about it:
1. **Root cause fix**: `pipeline.py` previously uploaded the raw PDF directly, which routes it through OGX's server-side `inline::auto` → `pypdf` text extraction — a completely different, uncontrollable extraction path from the one `import.py` uses (client-side, via `docling`/`make ingest-data`). Switched `pipeline.py` to upload the pre-converted `.md` instead, avoiding OGX's server-side PDF extraction entirely. Confirmed live: the exact query that previously returned glyph noise from the PDF-sourced vector store returns clean table text from the markdown-sourced one.
2. **Durable, document-agnostic guard**: `import.py`'s `redact_noisy_sections()` (duplicated inline in `pipeline.py`'s KFP component — see above) scans text for the *shape* of OCR/extraction noise (not this document's specific words) and replaces detected spans with an explicit "content omitted" note, so future re-OCRs/re-extractions of this or any other document degrade visibly instead of silently indexing garbage. It's a two-stage detector — a coarse sliding-window scan for recall, then a per-line trim from both ends for precision — because neither alone was good enough: the coarse scan alone bled redaction into 18 legitimate spec-table rows in testing, and a naive "doesn't look like real content" broad classifier flagged a legitimate parts-catalog list (short abbreviations, OCR'd fraction notation) as 80%+ noise. See the long comment block above `noise_ratio()` in `import.py` for the full calibration notes.

The original ~6 garbled regions in `chatbot/src/assets/c3_repair.md` were found and fixed by hand (cross-checked against the rendered page images in `target/data/c3_repair_artifacts/` from a local `make ingest-data` run) before `redact_noisy_sections()` existed; running the finished detector against that same original (pre-fix) text independently found the same regions plus one more (margin-noise near "GENERAL INFORMATION AND MAINTENANCE") that the by-hand pass had missed — that one's since been fixed the same way, via the automated tool.

A separate, lower-severity issue remains unfixed: a few regions (e.g. around the piston-ring-gap/torque-spec tables) have real data reflowed one-token-per-line due to a multi-column-page reading-order failure, not character-level corruption — the numbers are present, just not reconstructable without manual cross-referencing against the source page images. `redact_noisy_sections()` deliberately does not touch these (the real content stays indexed, even if poorly formatted) to stay on the high-precision side of the trade-off.

**`corvetteforum-mcp/`** — MCP server (`src/app.py`)
- Exposes a single tool `search_c3_tech_support` via FastMCP/SSE.
- Uses Google Custom Search API (requires `GCP_API_KEY` and `GOOGLE_CX` env vars) to search corvetteforum.com, then fetches the top result and converts HTML to markdown via `docling`.
- Runs on port 8080 by default (`MCP_PORT` env var to override).

### Infrastructure

- **OGX** (formerly Llama Stack): OpenAI-compatible API layer sitting in front of the LLM (vLLM) and Milvus. Deployed on OpenShift AI using the OGX Operator via the `OGXServer` CRD (`deploy/ocp/ogx_deployment_using_operator.yaml`, `deploy/helm/templates/ogx_deployment.yaml`).
- **Helm chart** (`deploy/helm/`): Deploys the chatbot to OpenShift. Image pulled from `registry.home.glroland.com/mechanic/`. Install with `helm install m1 .` from `deploy/helm/`.
- **CI/CD**: `deploy/Jenkinsfile` builds the chatbot Docker image and pushes to the registry.
- **Vector DB name**: hardcoded as `mechanic_vector_db` in both `import.py` and `ai_gateway.py`.
- **MLflow GenAI tracing**: logs traces to the MLflow server running on OpenShift AI, same setup as the `distribution-center` project. `MLFLOW_TRACKING_AUTH=kubernetes-namespaced` (set via the chatbot's ConfigMap) auto-reads the pod's own service-account token for auth — this requires `deploy/helm/templates/serviceaccount-mlflow.yaml`'s `RoleBinding` to the `mlflow-operator-mlflow-integration` ClusterRole, and `mlflow[kubernetes]` in `chatbot/requirements.txt`. Locally, `make run-chatbot` instead passes a raw `MLFLOW_TRACKING_TOKEN` (from `oc whoami --show-token`), since there's no pod/namespace to derive kubernetes-namespaced auth from outside the cluster.

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

**Diagnosed 2026-10-21, a second, separate failure** after the above was deployed: `PermissionError: [Errno 13] Permission denied: '/opt/app-root/src/.ogx/distributions/rh/files'` on startup. Confirmed via `oc debug` against the actual crashing pod in the `mechanic` namespace that the mount path itself wasn't the root cause — `/opt/app-root/src/.ogx` (the PVC root) was `777 nobody:nobody`, fine, but the nested `distributions` and `distributions/rh` directories inside it were `755`+sticky, owned `root:root`, dated ~3 months before this migration (almost certainly left over from a prior root-run pod, e.g. under this project's `serviceaccount-anyuid.yaml` SCC). The pod now runs as a non-root uid (1001, `fsGroup: 1001`), which can't write into a directory only its owning uid (0) has write on.

First fix attempt (renaming the hand-rolled PVC and moving its `workload.overrides.volumes/volumeMounts` mount to `/.ogx`) hit a second problem live: **the OGX Operator always creates its own `emptyDir` named `ogx-storage` mounted at `/.ogx` by default**, regardless of what `workload.storage` is set to — confirmed by inspecting the operator-managed Deployment directly (`oc get deployment ... -o jsonpath='{.spec.template.spec.volumes}'` showed both an `emptyDir` named `ogx-storage` *and* our hand-rolled PVC volume, both targeting `/.ogx`, which the operator's reconciliation then rejected as `duplicate entries for key [mountPath="/.ogx"]`). The actual fix: use the structured `spec.workload.storage: {mountPath, size}` field instead of a hand-rolled `overrides.volumes/volumeMounts` PVC mount — that tells the operator to swap its own default `emptyDir` for a real, operator-managed PVC at the same path (named `<OGXServer-name>-pvc`), rather than fighting it with a second volume. No PVC template needed in the chart any more; `deploy/helm/templates/pvc.yaml` was deleted.

One more wrinkle: switching an *existing* OGXServer from the old hand-rolled-PVC shape to `workload.storage` makes the operator try to merge-patch the live Deployment through an incompatible volume-type change (`emptyDir` → PVC) in one step, which the Kubernetes API rejects (`may not specify more than 1 volume type`, plus a `strategy.rollingUpdate`/`Recreate` conflict). Deleting the operator-owned Deployment (not the OGXServer CR) let it reconcile fresh with no stale fields to conflict with.

The old `mechanic-pvc` claim (root-owned stale data) is not deleted automatically — it's unreferenced now and safe to remove whenever convenient.

**Diagnosed 2026-10-21, a third, separate failure** after the above was fixed: `ValueError: Provider 'sentence-transformers' not found` at startup. The embedding model was being registered (correctly, from `EMBEDDING_MODEL`/`EMBEDDING_PROVIDER_MODEL_ID` env vars) against a `provider_id: sentence-transformers` that the env-var-driven config generator never actually creates — confirmed by grepping both OGX operator binaries (`ogx-k8s-operator-controller-manager`, `opendatahub-ogx-operator` in `redhat-ods-applications`) and the server image itself for `ENABLE_SENTENCE_TRANSFORMERS`/`SENTENCE_TRANSFORMERS`: zero matches anywhere. The upstream `red-hat-data-services/ogx-distribution` repo's README confirms why: `inline::sentence-transformers` (along with `inline::milvus` and `inline::faiss`) is "stripped from the default `config.yaml`" for this distribution, though the provider plugin remains installed. Red Hat's own 3.5 demo articles configure embeddings exclusively via a **remote** vLLM-served model (`EMBEDDING_MODEL`/`EMBEDDING_PROVIDER_MODEL_ID`/`VLLM_EMBEDDING_URL`/`VLLM_EMBEDDING_TLS_VERIFY`) — there is no env var that enables the inline provider.

Tried `spec.providers.inference.inline.custom` (the CRD's structured provider field) first — this triggers a completely different "declarative config generation" code path that needs the operator to fetch OCI labels from the distribution image's registry manifest to resolve a base config, which failed with HTTP 401 against `registry.redhat.io` (no registry credentials available to the operator for that specific lookup, independent of the pod's own working image pull secret). Reverted.

Fixed instead with `spec.overrideConfig` (`deploy/helm/templates/ogx-config.yaml`, referenced from `ogx_deployment.yaml`): a full hand-written `config.yaml` that replaces the env-var-driven generator entirely. Built by taking the exact "Stack Configuration" block this OGXServer's own startup logs were already printing (so every value matches what the generator was correctly producing from `workload.overrides.env`) and adding one missing block — the `inline::sentence-transformers` provider — copied verbatim from the one other working OGXServer on this cluster (`sandbox` namespace, `lsd-genai-playground`'s `llama-stack-config` ConfigMap), which solves this exact problem the same way. `${env.VAR_NAME}`/`${env.VAR_NAME:=default}` substitution inside the ConfigMap still resolves from `workload.overrides.env`, confirmed against that same reference ConfigMap, so `VLLM_URL`/`MILVUS_*`/`POSTGRES_*` changes don't require editing the override file. Once `overrideConfig` is set, `ENABLE_SENTENCE_TRANSFORMERS`/`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`EMBEDDING_PROVIDER_MODEL_ID`/`ENABLE_INLINE_MILVUS` env vars do nothing — removed from `ogx_deployment.yaml`.

One easy mistake made along the way, in case you hit it again: the `vector_stores.default_embedding_model.model_id` in the override config must exactly match a registered model's bare `model_id` (`granite-embedding-125m-english`), not the `provider_model_id` (`ibm-granite/granite-embedding-125m-english`) — using the latter fails fast with a clear error listing the actual available embedding model IDs.

### Key Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENAI_BASE_URL` | chatbot, import.py | OGX endpoint URL (e.g. `http://localhost:8321/v1/`) |
| `OPENAI_API_KEY` | chatbot, import.py | API key (any non-empty value works with OGX) |
| `MODEL` | chatbot | LLM model name (e.g. `together/openai/gpt-oss-120b`) |
| `GCP_API_KEY` | corvetteforum-mcp | Google Custom Search API key |
| `GOOGLE_CX` | corvetteforum-mcp | Google Custom Search engine ID |
| `MLFLOW_TRACKING_URI` | chatbot (`tracing.py`) | MLflow server URL; unset disables tracing entirely |
| `MLFLOW_WORKSPACE` | chatbot (read natively by mlflow) | MLflow workspace (OpenShift AI namespace tracing logs to) |
| `MLFLOW_TRACKING_AUTH` | chatbot (read natively by mlflow) | `kubernetes-namespaced` in-cluster; unset locally (uses `MLFLOW_TRACKING_TOKEN` instead) |
| `MLFLOW_EXPERIMENT_NAME` | chatbot (read natively by mlflow) | MLflow experiment traces are grouped under |

### Python Version

Python 3.12 (matches the Docker base image `ubi9/python-312`).
