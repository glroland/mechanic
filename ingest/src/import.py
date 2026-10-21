""" CLI for loading content into vector database. """
import os
import re
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

# --- OCR/extraction noise detection -----------------------------------
#
# Generic, document-agnostic guard against the kind of corruption found (and
# manually removed) in chatbot/src/assets/c3_repair.md: OCR/extraction
# failures that emit garbage tokens in place of real table/paragraph content.
# This isn't specific to that one PDF - it runs on whatever markdown text
# it's given and looks for the *shape* of extraction noise, not this
# document's words, so it applies the same way to a different manual re-run
# through docling, or a re-OCR of this same PDF with different settings.
#
# Two stages, because neither alone was precise+recall-complete enough
# (calibrated 2026-10-21 against this document's actual garbled regions,
# pulled back out of git history, plus a deliberately adversarial set of
# legitimate dense reference content - a parts-catalog numbered list, an
# axle code table with dot-leaders, a garbled-but-still-readable OCR'd prose
# passage - chosen specifically to try to trip false positives before
# trusting this to run unsupervised):
#
#   1. COARSE DETECTION (noise_ratio(), a sliding window over NOISE_SCAN_*
#      lines): good recall, including the sparsest corruption pattern seen
#      here (garbage emitted one token per line, so a run needs ~50 lines to
#      accumulate enough tokens to be statistically meaningful) - but a wide
#      window bleeds noise-window flagging up to a window's width into real
#      content past the true boundary, which first-draft testing caught
#      eating 18 legitimate tune-up spec rows this document still has.
#   2. FINE TRIM (line_is_noisy(), per physical line): tight boundaries -
#      judges each line on its own content, so it can't bleed into a
#      neighboring table/paragraph the way a window can - but alone it
#      misses the one-token-per-line pattern, since an isolated 2+ char
#      fragment like "CM" or "Q." doesn't match the narrow noise-token
#      pattern on its own and there's no neighboring context to aggregate.
#
# Stage 1 finds the approximate region; stage 2 trims it from both ends down
# to just past the last/first line with direct noise evidence. Validated
# against the adversarial clean set with zero false positives, and finds
# all of this document's known-garbled regions (plus one small one missed
# during the original by-hand pass) within a line or two of the by-hand
# boundaries.
NOISE_MIN_TOKENS = 25

# A token that's a single bare letter/digit floating alone, or one of the
# specific repeated OCR-garbage-glyph shapes seen in this document's
# corruption ("o"/"d"/"CO"/"§" etc. in place of real digits/symbols). Real
# prose and part/spec codes essentially never tokenize down to a lone
# character - words are 2+ letters, codes are 2+ chars - so this is a
# narrow, high-precision signal. An earlier, broader classifier (accepting
# any token that didn't match a whitelist of "looks like real content"
# patterns) scored a legitimate parts-catalog list as 80%+ noise - short
# abbreviations ("Rr.", "Otr.", "Bshg") and OCR'd fraction notation
# ("7/16"-20x1%") don't match a reasonable whitelist, so "assume noise
# unless proven signal" is the wrong default; "assume signal unless it's
# this one specific narrow noise shape" isn't.
_NOISE_TOKEN_RE = re.compile(r"^([A-Za-z]|[oOdD0]{2,3}|CO|[§•®«»^])$")

# Markdown table syntax / bare punctuation - not a content token either way.
_PUNCT_ONLY_TOKEN_RE = re.compile(r"^[|:\-=#*!\[\]().,/\\'\"]+$")

# Used only for short (<= SPARSE_MAX_TOKENS-token) lines in line_is_noisy():
# a real word, a plain/decimal number, or an alphanumeric spec/part code
# (e.g. "R45TS", "27A"). Deliberately not used as the primary classifier
# (see the false-positive note above) - only to decide "does this short,
# otherwise-unclassifiable line contain at least one recognizable thing".
_SPARSE_SIGNAL_TOKEN_RE = re.compile(
    r"^[A-Za-z]{3,}$"
    r"|^\d+([.,]\d+)?$"
    r"|^[A-Za-z]{1,4}\d{1,5}[A-Za-z0-9]{0,4}$"
    r"|^\d{1,4}[A-Za-z]{1,4}\d{0,4}$"
)

# Stage 1 (coarse): sliding window wide enough to accumulate NOISE_MIN_TOKENS
# even in the sparsest known corruption pattern (~1 token per 2 physical
# lines), stepped finer than the window so a region doesn't have to align
# with a window boundary to be caught.
NOISE_SCAN_WINDOW_LINES = 40
NOISE_SCAN_STEP_LINES = 10
NOISE_SCAN_MERGE_GAP_LINES = 12
NOISE_RATIO_THRESHOLD = 0.20

# Stage 2 (fine trim): a line with more than this many tokens is judged as a
# whole (most-tokens-are-noise); a line with this many or fewer is judged by
# whether it contains any recognizable content at all (see
# _SPARSE_SIGNAL_TOKEN_RE above) - the one-token-per-line pattern only shows
# up at this granularity.
DENSE_LINE_NOISE_RATIO_THRESHOLD = 0.5
SPARSE_MAX_TOKENS = 5
# Trim stops (keeps the boundary) once a small lookahead/lookbehind window
# contains no noisy line - i.e. the boundary can sit up to this many lines
# past the last directly-confirmed-noisy line. Larger values trim less
# aggressively (more real content kept, but spans can bleed a bit further);
# smaller values risk splitting one noisy run into pieces if it has a short
# run of coincidentally-clean-looking lines in the middle.
TRIM_LOOKAHEAD_LINES = 3

# A trimmed span shorter than this is almost certainly a stage-1 false
# start (the coarse window found *something*, but there was nothing left
# after trimming) rather than real corruption, so it's dropped.
MIN_FINAL_SPAN_LINES = 4

NOISE_REDACTION_NOTE = (
    "> **[Content omitted: automated OCR/extraction quality check]** This "
    "section looked like extraction noise (garbled characters in place of "
    "real content) rather than real manual text, and was redacted before "
    "indexing. If this looks wrong, check the noise-detection thresholds in "
    "ingest/src/import.py / ingest/src/pipeline.py against the source page."
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s|]+", text) if t.strip()]


def noise_ratio(text: str) -> float:
    """ Fraction of a text sample's meaningful tokens that look like
        OCR/extraction noise. Stage 1 (coarse) classifier - see
        NOISE_RATIO_THRESHOLD and the module docstring above.
    """
    noise = total = 0
    for token in _tokens(text):
        if _PUNCT_ONLY_TOKEN_RE.fullmatch(token):
            continue
        total += 1
        if _NOISE_TOKEN_RE.fullmatch(token):
            noise += 1
    if total < NOISE_MIN_TOKENS:
        return 0.0
    return noise / total


def line_is_noisy(line: str) -> bool:
    """ Stage 2 (fine) classifier - judges a single physical line. See the
        module docstring above for why this needs two different rules
        depending on line length.
    """
    if line.strip().startswith("!["):
        return False  # markdown image reference - structural, not content
    tokens = [t for t in _tokens(line) if not _PUNCT_ONLY_TOKEN_RE.fullmatch(t)]
    if not tokens:
        return False
    if len(tokens) <= SPARSE_MAX_TOKENS:
        return not any(_SPARSE_SIGNAL_TOKEN_RE.match(t) for t in tokens)
    noisy = sum(1 for t in tokens if _NOISE_TOKEN_RE.fullmatch(t))
    return noisy / len(tokens) >= DENSE_LINE_NOISE_RATIO_THRESHOLD


def _find_coarse_spans(lines: list[str]) -> list[tuple[int, int]]:
    """ Stage 1: sliding-window scan, returns approximate (start, end)
        line-index spans (0-indexed, inclusive) that need trimming.
    """
    n = len(lines)
    flagged = [False] * n
    for start in range(0, n, NOISE_SCAN_STEP_LINES):
        end = min(start + NOISE_SCAN_WINDOW_LINES, n)
        if noise_ratio("\n".join(lines[start:end])) >= NOISE_RATIO_THRESHOLD:
            for i in range(start, end):
                flagged[i] = True

    spans = []
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        span_end = i
        gap = 0
        j = i
        while j < n:
            if flagged[j]:
                span_end = j
                gap = 0
            else:
                gap += 1
                if gap > NOISE_SCAN_MERGE_GAP_LINES:
                    break
            j += 1
        spans.append((i, span_end))
        i = span_end + 1
    return spans


def _trim_span(lines: list[str], a: int, b: int) -> tuple[int, int]:
    """ Stage 2: trims a coarse span inward from both ends down to just past
        the outermost line with direct per-line noise evidence.
    """
    def has_noisy_line(start, end):
        return any(line_is_noisy(l) for l in lines[start:end])

    while a <= b:
        if has_noisy_line(a, min(a + TRIM_LOOKAHEAD_LINES, b + 1)):
            break
        a += 1
    while b >= a:
        if has_noisy_line(max(b - TRIM_LOOKAHEAD_LINES + 1, a), b + 1):
            break
        b -= 1
    return a, b


def find_noisy_spans(text: str) -> list[tuple[int, int]]:
    """ Runs both detection stages over a whole document's lines. Returns
        final (start, end) line-index spans (0-indexed, inclusive).
    """
    lines = text.splitlines()
    spans = []
    for a0, b0 in _find_coarse_spans(lines):
        a, b = _trim_span(lines, a0, b0)
        if a <= b and b - a + 1 >= MIN_FINAL_SPAN_LINES:
            spans.append((a, b))
    return spans


def redact_noisy_sections(text: str) -> tuple[str, int]:
    """ Finds noisy spans (see find_noisy_spans()) and replaces each with
        NOISE_REDACTION_NOTE. Returns (cleaned_text, redacted_span_count).

        This is a line-range redaction, not a chunk-drop: it only removes
        the noisy span itself, preserving any real content before/after it
        in the same section/chunk. Boundaries are approximate (calibrated
        against known examples, not per-page image inspection) and may
        bleed a line or two of real content into a redacted span - an
        accepted tradeoff for running unsupervised on every ingest, instead
        of leaving known-garbled content indexed.
    """
    lines = text.splitlines()
    spans = find_noisy_spans(text)
    if not spans:
        return text, 0

    cleaned_lines = []
    cursor = 0
    for a, b in spans:
        cleaned_lines.extend(lines[cursor:a])
        cleaned_lines.append(NOISE_REDACTION_NOTE)
        cursor = b + 1
    cleaned_lines.extend(lines[cursor:])
    return "\n".join(cleaned_lines), len(spans)

class ErrorCodes:
    SUCCESS = 0
    ILLEGAL_ARGS = 1
    FILE_NOT_FOUND = 2
    OGX_CONFIG_ERROR = 3
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
        vdb_provider - vector db provider in lls
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

    # Connect to the OGX server (formerly Llama Stack)
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

    # Redact OCR/extraction-noise regions before chunking, so a corrupted
    # table/section gets replaced in place rather than taking out an entire
    # header-chunk around it. See redact_noisy_sections().
    file_contents, redacted_spans = redact_noisy_sections(file_contents)
    if redacted_spans:
        logger.warning("Redacted %s noisy section(s) from input file.", redacted_spans)

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
            'static': {
                'max_chunk_size_tokens': 1000,  # Set maximum chunk size to 1000 tokens
                'chunk_overlap_tokens': 200     # Set chunk overlap to 200 tokens
            }
        },
        extra_body={
            "provider_id": vdb_provider,
            "embedding_model": embedding_model_name,
#            "embedding_dimension": 384
        }
    )
    vector_store_id = new_vdb.id
    logger.info("Vector Database Created.  New VDB ID = %s", vector_store_id)

    # insert each chunk, skipping any that *still* look like mostly OCR/
    # extraction noise after redact_noisy_sections() above already redacted
    # noisy spans in place - a backstop for the rare chunk that's noise
    # through-and-through (so redaction left nothing but the note plus a
    # sliver of real content) rather than noise embedded in a larger
    # otherwise-good section.
    skipped = 0
    for i, chunk in enumerate(text_chunks):
        # upload chunk and insert into vector store
        #logger.info("Chunk: %s", chunk)
        chunk_str = str(chunk)

        chunk_noise = noise_ratio(chunk_str)
        if chunk_noise >= NOISE_RATIO_THRESHOLD:
            skipped += 1
            logger.warning(
                "Skipping chunk %s - still looks like OCR/extraction noise "
                "(%.0f%% of tokens) after redaction. First 100 chars: %r",
                i, chunk_noise * 100, chunk_str[:100]
            )
            continue

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

    if skipped:
        logger.warning("Skipped %s of %s chunks as noise.", skipped, len(text_chunks))

    # List files in a vector store
    vector_store_files = openai_client.vector_stores.files.list(vector_store_id=vector_store_id)
    logger.info("Vector Store File List: %s", vector_store_files)

    # Successfully imported
    logger.info("Successfully imported content.")
    sys.exit(ErrorCodes.SUCCESS)


if __name__ == '__main__':
    cli()
