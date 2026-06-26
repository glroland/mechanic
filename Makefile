OPENAI_BASE_URL := http://mechanic-openai-responses-api-service:8321/v1
OPENAI_API_KEY := nokeyneeded
MODEL := vllm-inference/gemma-4
EMBEDDING_MODEL := sentence-transformers/nomic-ai/nomic-embed-text-v1.5
VECTORDB_PROVIDER := milvus

MLFLOW_TRACKING_URI := https://rh-ai.apps.ocp.home.glroland.com/mlflow
MLFLOW_WORKSPACE := mechanic
MLFLOW_EXPERIMENT_ID := 3
MLFLOW_TRACKING_TOKEN := $(shell oc whoami --show-token)

LOG_LEVEL := DEBUG

HAS_UV := $(shell command -v uv >/dev/null 2>&1; if [ $$? -eq 0 ]; then echo "true"; else echo "false"; fi)
ifeq ($(HAS_UV), true)
    PIP = uv pip
else
    PIP = pip
endif

install:
	$(PIP) install -r chatbot/requirements.txt
	$(PIP) install -r corvetteforum-mcp/requirements.txt
	$(PIP) install -r ingest/requirements.txt

clean:
	rm -rf target

ingest-data:
	mkdir -p target/data
	cd target/data && docling --from pdf --to json --to md --image-export-mode referenced --ocr --output . --abort-on-error ../../chatbot/src/assets/c3_repair.pdf
	cd ingest/src && python import.py $(OPENAI_BASE_URL) $(EMBEDDING_MODEL) mechanic_vector_db ../../target/data/c3_repair.md

run-chatbot:
	@echo MLFlow Tracking URI: $(MLFLOW_TRACKING_URI)
	@echo
	@echo MLFlow Token: $(MLFLOW_TRACKING_TOKEN)
	@echo
	cd chatbot/src && MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) MLFLOW_WORKSPACE=$(MLFLOW_WORKSPACE) MLFLOW_EXPERIMENT_ID=$(MLFLOW_EXPERIMENT_ID) MLFLOW_TRACKING_TOKEN=$(MLFLOW_TRACKING_TOKEN) OPENAI_BASE_URL=$(OPENAI_BASE_URL) OPENAI_API_KEY=$(OPENAI_API_KEY) MODEL=$(MODEL) LOG_LEVEL=$(LOG_LEVEL) streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8080

run-corvetteforummcp:
	cd corvetteforum-mcp/src && python app.py

compile-pipeline:
	cd ingest/src && python pipeline.py

vectorize:
	cd ingest/src && OPENAI_API_KEY=$(OPENAI_API_KEY) python import.py $(OPENAI_BASE_URL) $(EMBEDDING_MODEL) $(VECTORDB_PROVIDER) ../../chatbot/src/assets/c3_repair.md
