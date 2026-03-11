OPENAI_BASE_URL := https://my-llama-stack-my-llama-stack.apps.ocp.home.glroland.com/v1/
MODEL := together/openai/gpt-oss-120b
EMBEDDING_MODEL := sentence-transformers/sentence-transformers/all-mpnet-base-v2
VECTORDB_PROVIDER := milvus

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
	cd chatbot/src && OPENAI_BASE_URL=$(OPENAI_BASE_URL) OPENAI_API_KEY=$(OPENAI_API_KEY) MODEL=$(MODEL) streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8080

run-corvetteforummcp:
	cd corvetteforum-mcp/src && python app.py

compile-pipeline:
	cd ingest/src && python pipeline.py

test:
	cd ingest/src && python import.py $(OPENAI_BASE_URL) $(EMBEDDING_MODEL) $(VECTORDB_PROVIDER) ../../target/data/c3_repair.md
