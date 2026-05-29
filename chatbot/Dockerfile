FROM registry.access.redhat.com/ubi9/python-312:9.6-1760372467

# Configuration
ENV LOG_LEVEL = "INFO"
ENV LLAMA_STACK_URL "http://localhost:8321"
ENV EMBEDDING_MODEL "sentence-transformers/all-mpnet-base-v2"
ENV API_KEY = "nokeyneeded"
ENV MODEL = "openai/gpt-4"
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENV OTEL_SERVICE_NAME=mechanic-chatbot
ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=false
ENV OTEL_EXPORTER_OTLP_HEADERS=""
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_AUTH=""
#ENV MLFLOW_TRACKING_TOKEN=""
ENV MLFLOW_TRACKING_INSECURE_TLS="true"
#ENV MLFLOW_TRACKING_HEADERS='{"x-mlflow-workspace": "your-project-namespace"}'
ENV MLFLOW_ENABLE_ASYNC_TRACE_LOGGING="true"
ENV MLFLOW_TRACE_SAMPLING_RATIO="1.0"
ENV MLFLOW_TRACE_TIMEOUT_SECONDS="30"
#ENV MLFLOW_EXPERIMENT_ID=""
#ENV MLFLOW_WORKSPACE=""
ENV DISABLE_MLFLOW_STATS="true"
#ENV SET_TOKEN_FROM_FILE=""

# By default, listen on port 8080
EXPOSE 8080/tcp

# Set the working directory in the container
WORKDIR /projects

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./src/ .

# Specify the command to run on container start
CMD ["streamlit", "run", "./app.py", "--server.port=8080", "--server.address=0.0.0.0"]
