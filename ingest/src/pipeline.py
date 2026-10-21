from kfp import compiler, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json

# Workbench Runtime Image: Pytorch with CUDA and Python 3.12 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.23.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    # Llama Stack was renamed to OGX in OpenShift AI 3.5. llama-stack-client's
    # successor is "ogx-client" on PyPI (verified 2026-10-21 by installing it
    # in a scratch venv and reading its generated source/pydantic schemas -
    # no live OGX server was reachable from that sandbox to round-trip an
    # actual HTTP call, so this is confirmed at the request/response-shape
    # level, not end-to-end). It's a generated client (ogx_client.OgxClient),
    # but it's still the ergonomic one: client.files / client.models /
    # client.embeddings / client.vector_stores / client.vector_stores_files
    # all exist and accept the same kwargs/dict style used below - its
    # "WithExtraBody" request models explicitly route any unrecognized kwarg
    # (including ones passed via extra_body=) into their additional_properties
    # field, so embedding_model/embedding_dimension/provider_id survive
    # unchanged. Two real differences from llama_stack_client:
    #   - client.vector_stores.files is now the top-level client.vector_stores_files
    #   - "timeout=" isn't a real kwarg on these calls; use "_request_timeout="
    # and client.models.list() returns a oneOf wrapper - unwrap via
    # ".actual_instance.data" to get the list of models.
    packages_to_install=["ogx-client==1.5.0", "fire", "requests"],
)
def register_vector_store_and_files(
    service_url: str,
    vector_store_name: str,
    embedding_model_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
    base_url: str,
    pdf_filenames: str,
    timeout: float,
):
    import io
    import requests
    from ogx_client import OgxClient

    client = OgxClient(base_url=service_url)

    # Upload all files first and collect file_ids
    file_ids = []
    for filename in pdf_filenames.split(","):
        source = f"{base_url}/{filename.strip()}"
        print("Downloading and uploading document:", source)

        try:
            # Download the docs from URL
            response = requests.get(source)
            response.raise_for_status()  # Raise an exception for bad status codes

            file_content = io.BytesIO(response.content)
            file_basename = source.split("/")[-1]

            # Upload file to storage
            file = client.files.create(
                file=(file_basename, file_content, "application/pdf"),
                purpose="assistants",
            )
            file_ids.append(file.id)
            print(f"Successfully uploaded {file_basename} (file_id: {file.id})")

        except Exception as e:
            print(f"ERROR: Failed to upload {filename.strip()}: {str(e)}")
            raise

    print(f"Successfully uploaded {len(file_ids)} files: {file_ids}")

    # client.models.list() returns a oneOf wrapper; .actual_instance is the
    # real OpenAIListModelsResponse with a .data list of OpenAIModel objects
    # (confirmed: OpenAIModel has the same id/custom_metadata fields the
    # native llama-stack Model type did).
    models = client.models.list().actual_instance.data
    matching_model = next((m for m in models if m.id == embedding_model_id), None)

    if not matching_model:
        available = [m.id for m in models]
        raise ValueError(
            f"Model '{embedding_model_id}' not found. Available: {available}"
        )

    model_type = (
        matching_model.custom_metadata.get("model_type")
        if matching_model.custom_metadata
        else None
    )
    if model_type != "embedding":
        raise ValueError(
            f"Model '{embedding_model_id}' is not an embedding model (type={model_type})"
        )

    embedding_dimension = int(
        float(matching_model.custom_metadata.get("embedding_dimension"))
    )

    # Warm up the embedding model
    client.embeddings.create(
        model=embedding_model_id,
        input="warmup",
        _request_timeout=timeout,
    )

    # Create empty vector store first, before inserting files.
    # Purpose: Depending on the size and number of files, attempting to create the vector store
    # and add files in a single step may lead to timeouts.
    try:
        vector_store = client.vector_stores.create(
            name=vector_store_name,
            file_ids=[],
            chunking_strategy={
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": max_tokens,
                    "chunk_overlap_tokens": chunk_overlap_tokens,
                },
            },
            extra_body={
                "embedding_model": embedding_model_id,
                "embedding_dimension": embedding_dimension,
                "provider_id": "milvus",
            },
            _request_timeout=timeout,
        )
        print(
            f"Successfully created vector store '{vector_store_name}' with ID: {vector_store.id}"
        )
    except Exception as e:
        print(f"ERROR: Failed to create vector store '{vector_store_name}': {str(e)}")
        raise

    # Add files to vector store
    try:
        for file_id in file_ids:
            print(f"Adding file_id '{file_id}' to vector store '{vector_store_name}'")
            # Note: this is client.vector_stores_files, not client.vector_stores.files -
            # ogx-client puts it at the top level, unlike llama_stack_client.
            client.vector_stores_files.create(
                vector_store_id=vector_store.id,
                file_id=file_id,
                _request_timeout=timeout,
            )
        vector_store = client.vector_stores.retrieve(vector_store.id)
        print(f"Vector store details: {vector_store}")
    except Exception as e:
        print(f"WARNING: Some files failed to be added to vector store: {str(e)}")


@dsl.pipeline()
def vector_store_files_pipeline(
    base_url: str = "https://raw.githubusercontent.com/glroland/mechanic/refs/heads/main/chatbot/src/assets",
    pdf_filenames: str = "c3_repair.pdf",
    vector_store_name: str = "mechanic_vector_db",
    service_url: str = "https://my-llama-stack-my-llama-stack.apps.ocp.home.glroland.com",
    embedding_model_id: str = "sentence-transformers/ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 2500,
    chunk_overlap_tokens: int = 64,
    use_gpu: bool = False,
    timeout: float = 5 * 60,
):
    """
    Creates a vector store with embeddings from PDF files from a GitHub source.
    :param base_url: Base URL to fetch PDF files from
    :param pdf_filenames: Comma-separated list of PDF filenames to download and convert
    :param vector_store_name: Name of the vector store to store embeddings
    :param service_url: URL of the Milvus service
    :param embedding_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :param chunk_overlap_tokens: Number of overlapping tokens between chunks
    :param use_gpu: Enable GPU usage for embedding generation
    :param timeout: Timeout to use for each OGX (formerly Llama Stack) API call
    :return:
    """

    with dsl.If(use_gpu == True):
        register_task = register_vector_store_and_files(
            service_url=service_url,
            vector_store_name=vector_store_name,
            embedding_model_id=embedding_model_id,
            max_tokens=max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            base_url=base_url,
            pdf_filenames=pdf_filenames,
            timeout=timeout,
        )
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")
        register_task.set_accelerator_type("nvidia.com/gpu")
        register_task.set_accelerator_limit(1)
        add_toleration_json(
            register_task,
            [
                {
                    "effect": "NoSchedule",
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                }
            ],
        )
        add_node_selector_json(register_task, {})

    with dsl.Else():
        register_task = register_vector_store_and_files(
            service_url=service_url,
            vector_store_name=vector_store_name,
            embedding_model_id=embedding_model_id,
            max_tokens=max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            base_url=base_url,
            pdf_filenames=pdf_filenames,
            timeout=timeout,
        )
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        vector_store_files_pipeline,
        package_path=__file__.replace(".py", "_compiled.yaml"),
    )
