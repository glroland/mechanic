from kfp import compiler, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json

# Workbench Runtime Image: Pytorch with CUDA and Python 3.12 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.23.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    # Llama Stack was renamed to OGX in OpenShift AI 3.5. llama-stack-client's
    # successor is "ogx-client" on PyPI - a generated client
    # (ogx_client.OgxClient), but still the ergonomic one: client.files /
    # client.models / client.embeddings / client.vector_stores /
    # client.vector_stores_files all exist and accept the same kwargs/dict
    # style used below - its "WithExtraBody" request models explicitly
    # route any unrecognized kwarg (including ones passed via extra_body=)
    # into their additional_properties field, so
    # embedding_model/embedding_dimension/provider_id survive unchanged.
    # Three real differences from llama_stack_client, the first two
    # confirmed only by reading the installed package (no live server
    # reachable at the time), the third confirmed against a real pipeline
    # run on 2026-10-21:
    #   - client.vector_stores.files is now the top-level client.vector_stores_files
    #   - "timeout=" isn't a real kwarg on these calls; use "_request_timeout="
    #   - every call below uses the "_without_preload_content" variant and
    #     reads .json() directly rather than the typed create()/list()/
    #     retrieve() methods, because ogx-client==1.5.0's bundled response
    #     models are stricter than what the actual OGX server version on
    #     this cluster (1.2.4+rhaiv.0) returns - e.g. OpenAIFileObject
    #     requires a "status" field client.files.create()'s response is
    #     missing, so it raised a pydantic ValidationError even though the
    #     upload itself succeeded (HTTP 200).
    packages_to_install=["ogx-client==1.5.0", "fire", "requests"],
)
def register_vector_store_and_files(
    service_url: str,
    vector_store_name: str,
    embedding_model_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
    base_url: str,
    source_filenames: str,
    timeout: float,
):
    import requests
    from ogx_client import OgxClient

    client = OgxClient(base_url=service_url)

    def _parse_response(response):
        """Read an ogx_client *_without_preload_content() response and return
        its parsed JSON. Confirmed live 2026-10-21: a naive
        response.json()["id"] on a non-2xx response raises a misleading
        KeyError (e.g. 'id') with the real server-side error message
        discarded - this surfaces that message instead.
        """
        response.read()
        if not response.is_success:
            raise RuntimeError(
                f"OGX request failed: {response.status_code} "
                f"{response.reason_phrase} - {response.text}"
            )
        return response.json()

    # Every call below uses the "_without_preload_content" variant and reads
    # .json() directly, instead of the typed create()/list()/retrieve()
    # methods. Confirmed live 2026-10-21: ogx-client==1.5.0's bundled
    # response models are stricter than what this OGX server (1.2.4+rhaiv.0)
    # actually returns - e.g. OpenAIFileObject requires a "status" field the
    # server's /v1/files response doesn't include, so client.files.create()
    # raised a pydantic ValidationError even though the upload itself
    # succeeded (HTTP 200). The "_without_preload_content" variants return
    # the raw httpx.Response and skip that validation entirely.
    #
    # ogx_client always sends requests with stream=True (confirmed in
    # ogx_client.rest.RESTClientObject.request), so every response here is
    # unread until .read() is called explicitly - the typed create() methods
    # do this internally before deserializing (hence "RESTResponse.read()
    # must be called before passing it to response_deserialize()" in
    # ApiClient.response_deserialize's own assert message), but the
    # "_without_preload_content" variants skip that step entirely. Calling
    # .json()/.content on an unread httpx.Response raises
    # httpx.ResponseNotRead - confirmed live 2026-10-21. Call .read() first.

    # Upload all files first and collect file_ids
    #
    # Source files are the already-converted markdown (chatbot/src/assets/*.md),
    # not the raw PDF. Confirmed live 2026-10-21 (MLflow trace
    # tr-e88efb22a600a006273afff02d0aa8e1, mlflow.traceInputs): uploading the raw
    # PDF here routes it through OGX's inline::auto file_processors provider,
    # which does its own server-side pypdf text extraction. For this particular
    # PDF (an old Word doc run through "Acrobat PDFWriter 4.05", per its own
    # embedded metadata), several tables use a font whose glyph encoding is
    # corrupted in the source file itself - confirmed by rendering those pages
    # to an image (not just reading the text layer) and seeing the same garbage
    # glyphs, so no extraction method can recover them. pypdf hits this on every
    # affected page; docling (used to produce the .md file, see `make
    # ingest-data`) was run once, by hand, with its failures manually cleaned up
    # in the committed .md (see git history for chatbot/src/assets/c3_repair.md).
    # Uploading that already-clean markdown instead avoids re-running (and
    # re-breaking on) server-side PDF extraction altogether - confirmed live by
    # uploading a .md file through this same endpoint and observing clean
    # search results for a query that previously returned glyph noise from the
    # PDF-sourced vector store.
    file_ids = []
    for filename in source_filenames.split(","):
        source = f"{base_url}/{filename.strip()}"
        print("Downloading and uploading document:", source)

        try:
            # Download the docs from URL
            response = requests.get(source)
            response.raise_for_status()  # Raise an exception for bad status codes

            file_basename = source.split("/")[-1]

            # ogx_client.ApiClient.files_parameters() only accepts a
            # (filename, filedata) 2-tuple of raw bytes - no content-type as
            # a 3rd element (unlike llama_stack_client/the OpenAI SDK
            # convention) and it does `filename, filedata = v` directly with
            # no .read(), so a BytesIO wrapper isn't safe either.
            # Content-type gets auto-guessed from the filename extension via
            # mimetypes.guess_type() - resolves to "text/markdown" for the
            # .md sources used here (confirmed: mimetypes already registers
            # that extension, no explicit override needed).
            upload_response = client.files.create_without_preload_content(
                file=(file_basename, response.content),
                purpose="assistants",
            )
            file_id = _parse_response(upload_response)["id"]
            file_ids.append(file_id)
            print(f"Successfully uploaded {file_basename} (file_id: {file_id})")

        except Exception as e:
            print(f"ERROR: Failed to upload {filename.strip()}: {str(e)}")
            raise

    print(f"Successfully uploaded {len(file_ids)} files: {file_ids}")

    # client.models.list() just proxies to client.openai.list(); calling the
    # latter's raw variant directly to skip its oneOf-wrapper deserialization.
    models = _parse_response(client.openai.list_without_preload_content())["data"]
    matching_model = next((m for m in models if m["id"] == embedding_model_id), None)

    if not matching_model:
        available = [m["id"] for m in models]
        raise ValueError(
            f"Model '{embedding_model_id}' not found. Available: {available}"
        )

    custom_metadata = matching_model.get("custom_metadata") or {}
    model_type = custom_metadata.get("model_type")
    if model_type != "embedding":
        raise ValueError(
            f"Model '{embedding_model_id}' is not an embedding model (type={model_type})"
        )

    embedding_dimension = int(float(custom_metadata.get("embedding_dimension")))

    # Warm up the embedding model. Response is discarded, but still parsed
    # (rather than left unread) so a failure here raises a clear error
    # instead of leaving the httpx streaming response open.
    _parse_response(
        client.embeddings.create_without_preload_content(
            model=embedding_model_id,
            input="warmup",
            _request_timeout=timeout,
        )
    )

    # Create empty vector store first, before inserting files.
    # Purpose: Depending on the size and number of files, attempting to create the vector store
    # and add files in a single step may lead to timeouts.
    try:
        create_response = client.vector_stores.create_without_preload_content(
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
                # Must match the vector_io provider_id actually registered
                # on the server (deploy/helm/templates/ogx-config.yaml),
                # not just the provider_type - confirmed live 2026-10-21:
                # "milvus" (stale, pre-OGX-migration value) failed with
                # "Provider `milvus` not found"; the registered provider_id
                # is "milvus-remote".
                "provider_id": "milvus-remote",
            },
            _request_timeout=timeout,
        )
        vector_store_id = _parse_response(create_response)["id"]
        print(
            f"Successfully created vector store '{vector_store_name}' with ID: {vector_store_id}"
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
            _parse_response(
                client.vector_stores_files.create_without_preload_content(
                    vector_store_id=vector_store_id,
                    file_id=file_id,
                    _request_timeout=timeout,
                )
            )
        vector_store_details = _parse_response(
            client.vector_stores.retrieve_without_preload_content(vector_store_id)
        )
        print(f"Vector store details: {vector_store_details}")
    except Exception as e:
        print(f"WARNING: Some files failed to be added to vector store: {str(e)}")


@dsl.pipeline()
def vector_store_files_pipeline(
    base_url: str = "https://raw.githubusercontent.com/glroland/mechanic/refs/heads/main/chatbot/src/assets",
    # Pre-converted markdown, not the raw PDF - see the long comment in
    # register_vector_store_and_files() above for why: OGX's own server-side
    # PDF text extraction corrupts several tables in this specific document.
    source_filenames: str = "c3_repair.md",
    vector_store_name: str = "mechanic_vector_db",
    # Default assumes this pipeline runs in the same namespace as the
    # mechanic-openai-responses-api OGXServer (true for the mechanic DSPA) -
    # a bare Kubernetes Service name resolves within the caller's own
    # namespace via in-cluster DNS, so this doesn't need to hardcode
    # "mechanic" or any other namespace. Previously defaulted to
    # "https://my-llama-stack-my-llama-stack.apps.ocp.home.glroland.com", a
    # separate, unrelated dev environment (see helm-dev in README.md) - a
    # pipeline run against that default (confirmed live 2026-10-21) silently
    # succeeds through file upload/model list/embedding warmup against it,
    # then fails on vector store creation for reasons that have nothing to
    # do with this project's own OGX instance.
    service_url: str = "http://mechanic-openai-responses-api-service:8321",
    # Must be "<provider_id>/<model_id>" exactly as the server's own
    # registered_resources.models lists it, not provider_model_id - these
    # differ for this one model. Confirmed live 2026-10-21: the "ibm-granite/"
    # prefix here is this model's provider_model_id
    # (ibm-granite/granite-embedding-125m-english in
    # deploy/helm/templates/ogx-config.yaml's registered_resources.models),
    # not its model_id (granite-embedding-125m-english, no prefix) - the
    # OGXServer's own "Available" error list confirmed the registered id is
    # "sentence-transformers/granite-embedding-125m-english".
    embedding_model_id: str = "sentence-transformers/granite-embedding-125m-english",
    max_tokens: int = 2500,
    chunk_overlap_tokens: int = 64,
    use_gpu: bool = False,
    timeout: float = 5 * 60,
):
    """
    Creates a vector store with embeddings from pre-converted markdown files
    from a GitHub source.
    :param base_url: Base URL to fetch source files from
    :param source_filenames: Comma-separated list of filenames to download and upload.
        Must be pre-extracted text (markdown/.md), not raw PDFs - see the comment
        in register_vector_store_and_files() for why.
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
            source_filenames=source_filenames,
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
            source_filenames=source_filenames,
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
