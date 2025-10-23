FROM registry.access.redhat.com/ubi9/python-312:9.6-1760372467

# Configuration
ENV LLAMA_STACK_URL "http://localhost:8321"
ENV EMBEDDING_MODEL "sentence-transformers/all-mpnet-base-v2"
ENV API_KEY = "nokeyneeded"
ENV MODEL = "openai/gpt-4"

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
