OPENAI_BASE_URL := https://mechanic-openai-responses-api-mechanic.apps.ocp.home.glroland.com/v1/
MODEL := vllm-inference/gemma-4
EMBEDDING_MODEL := ibm-granite/granite-embedding-125m-english
VECTORDB_PROVIDER := milvus

MLFLOW_TRACKING_URI := https://data-science-gateway.apps.ocp.home.glroland.com/mlflow
MLFLOW_WORKSPACE := mechanic
MLFLOW_TRACKING_TOKEN := $(shell oc whoami --show-token)

OS := $(shell uname -s)

HAS_UV := $(shell command -v uv >/dev/null 2>&1; if [ $$? -eq 0 ]; then echo "true"; else echo "false"; fi)
ifeq ($(HAS_UV), true)
    PIP = uv pip
else
    PIP = pip
endif

install:
	$(PIP) install -r chatbot/requirements.txt
ifeq ($(OS),Darwin)
	$(PIP) install -r corvetteforum-mcp/requirements.txt.mac
	$(PIP) install -r ingest/requirements.txt.mac
else
	$(PIP) install -r corvetteforum-mcp/requirements.txt.linux
	$(PIP) install -r ingest/requirements.txt.linux
endif

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
	cd chatbot/src && MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) MLFLOW_WORKSPACE=$(MLFLOW_WORKSPACE) MLFLOW_TRACKING_TOKEN=$(MLFLOW_TRACKING_TOKEN) OPENAI_BASE_URL=$(OPENAI_BASE_URL) OPENAI_API_KEY=$(OPENAI_API_KEY) MODEL=$(MODEL) streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8080

run-corvetteforummcp:
	cd corvetteforum-mcp/src && python app.py

compile-pipeline:
	cd ingest/src && python pipeline.py

test:
	cd ingest/src && python import.py $(OPENAI_BASE_URL) $(EMBEDDING_MODEL) $(VECTORDB_PROVIDER) ../../target/data/c3_repair.md
