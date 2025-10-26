""" CLI for loading content into vector database. """
import os
import sys
import logging
from io import BytesIO
import click
from openai import OpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

MECHANIC_VECTOR_DB_NAME = "mechanic_vector_db"

ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

DEFAULT_TIMEOUT = 30 * 60

VECTOR_STORE_PROVIDER = "milvus-local"  # milvus-remote, milvus-local, faiss

class ErrorCodes:
    SUCCESS = 0
    ILLEGAL_ARGS = 1
    FILE_NOT_FOUND = 2
    LLS_CONFIG_ERROR = 3
    NO_API_KEY = 4

class ColorOutputFormatter(logging.Formatter):
    """ Add colors to stdout logging output to simplify text.  Thank you to:
        https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = '%(name)-13s: %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

@click.command()
@click.argument('openai_baseurl')
@click.argument('embedding_model_name')
@click.argument('vdb_provider')
@click.argument('input_file')
def cli(openai_baseurl: str, embedding_model_name: str, vdb_provider:str, input_file: str):
    """ CLI for importing mechanic content into vector store.
    
        openai_baseurl - OpenAI Base URL
        embedding_model_name - Embedding Model Name
        input_file - Input file to ingest
    """
    # Default to not set
    logging.getLogger().setLevel(logging.NOTSET)

    # Log info and higher to the console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(ColorOutputFormatter())
    logging.getLogger().addHandler(console)

    # Validate arguments
    if openai_baseurl is None or len(openai_baseurl) == 0:
        logger.fatal("OpenAI Base URL is a required parameter and cannot be empty.")
        sys.exit(ErrorCodes.ILLEGAL_ARGS)
    if embedding_model_name is None or len(embedding_model_name) == 0:
        logger.fatal("Embedding Model is a required parameter and cannot be empty.")
        sys.exit(ErrorCodes.ILLEGAL_ARGS)
    if vdb_provider is None or len(vdb_provider) == 0:
        logger.fatal("VectorDB Provider is required and cannot be empty.")
        sys.exit(ErrorCodes.ILLEGAL_ARGS)
    if input_file is None or len(input_file) == 0:
        logger.fatal("Input File is a required paramter and cannot be empty.")
        sys.exit(ErrorCodes.ILLEGAL_ARGS)

    # Ensure the input file exists
    if not os.path.exists(input_file):
        logger.fatal("Input File does not exist!  Filename = %s", input_file)
        sys.exit(ErrorCodes.FILE_NOT_FOUND)
    if not os.path.isfile(input_file):
        logger.fatal("Input File is a directory.  This is currently unsupported functionality.  Filename = %s", input_file)
        sys.exit(ErrorCodes.FILE_NOT_FOUND)

    # get openai key
    apikey = None
    if ENV_OPENAI_API_KEY in os.environ and len(os.environ[ENV_OPENAI_API_KEY]) > 0:
        apikey = os.environ[ENV_OPENAI_API_KEY]
    if apikey is None:
        logger.fatal("OpenAI API Key not specified!")
        sys.exit(ErrorCodes.NO_API_KEY)

    # Connect to LLama Stack
    logger.info("Connecting to OpenAI Compatible Endpoint.  URL=%s", openai_baseurl)
    openai_client = OpenAI(
        base_url=openai_baseurl,
        api_key=apikey,
        timeout=DEFAULT_TIMEOUT
    )
    logger.info("Successfully connected to OpenAI Compatible Endpoint.")

    # Load the input file
    file_contents = ""
    with open(input_file, "r") as f:
        file_contents = f.read()
    if len(file_contents) == 0:
        logger.fatal("Input file is empty.  Cannot load.")
        sys.exit(ErrorCodes.FILE_NOT_FOUND)
    logger.info("Loaded input file.  Size of File=%s", len(file_contents))

    # Split document into chunks
    logger.info("Chunking file...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_chunks = markdown_splitter.split_text(file_contents)
    logger.info("File Chunked.  # of Chunks = %s", len(text_chunks))

    # Create a list of the registred Vector Databases
    vector_stores = openai_client.vector_stores.list()
    logger.info("Searching list of Vector DBs for pre-existing repositories.  Pre-existing: %s", vector_stores)
    for vector_db in vector_stores:
        logger.debug("Vector DB in list.  ID=%s; Name=%s", vector_db.id, vector_db.name)
        if vector_db.name == MECHANIC_VECTOR_DB_NAME:
            logger.warning("Preexisting instance of the vector database.  Deleting....  id=%s", vector_db.id)
            openai_client.vector_stores.delete(vector_store_id = vector_db.id)

    # Register a vector database
    logger.info("Creating new Vector Database for content.  ID/Name=%s", MECHANIC_VECTOR_DB_NAME)
    new_vdb = openai_client.vector_stores.create(
        name=MECHANIC_VECTOR_DB_NAME,
        chunking_strategy={
            'type': 'static',
            #'static': {
            #    'max_chunk_size_tokens': 1000,  # Set maximum chunk size to 1000 tokens
            #    'chunk_overlap_tokens': 200     # Set chunk overlap to 200 tokens
            #}
        },
        extra_body={
            "provider_id": VECTOR_STORE_PROVIDER,
#            "embedding_model": embedding_model_name,
#            "embedding_dimension": 384
        }
    )
    vector_store_id = new_vdb.id
    logger.info("Vector Database Created.  New VDB ID = %s", vector_store_id)

    # insert each chunk
    for i, chunk in enumerate(text_chunks):
        # upload chunk and insert into vector store
        #logger.info("Chunk: %s", chunk)
        chunk_str = str(chunk)
        file_content_bytes = BytesIO(chunk_str.encode('utf-8'))
        uploaded_file = openai_client.files.create(
            file=(f"chunk-{i}.txt", file_content_bytes, "text/plain"),
            purpose="assistants"
        )
        logger.info("Uploaded Chunk as File: %s", uploaded_file.id)
        vector_store_file = openai_client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=uploaded_file.id
        )
        logger.info("Associated Uploaded File with Vector Store: %s", vector_store_file.id)

    # List files in a vector store
    vector_store_files = openai_client.vector_stores.files.list(vector_store_id=vector_store_id)
    logger.info("Vector Store File List: %s", vector_store_files)

    # Successfully imported
    logger.info("Successfully imported content.")
    sys.exit(ErrorCodes.SUCCESS)


if __name__ == '__main__':
    cli()
