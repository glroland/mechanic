""" Convert Document Example

Standalone demo page showing customers how a document is turned into the
markdown the ingestion pipeline chunks and uploads, using the same docling
library `make ingest-data` runs against the service manual PDF.
"""
import streamlit as st

from constants import AppUserInterfaceElements
from chrome import setup_example_page

card = setup_example_page(AppUserInterfaceElements.CONVERT_DOCUMENT_PAGE_TITLE, "📄")

with card:
    st.title(AppUserInterfaceElements.CONVERT_DOCUMENT_PAGE_TITLE)
    st.write("This is the same docling conversion `make ingest-data` runs "
             "against the service manual PDF, turning a document into the "
             "markdown that later gets chunked and indexed.")

    EXAMPLE_CODE = '''from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("service_manual.pdf")
markdown = result.document.export_to_markdown()
'''
    st.code(EXAMPLE_CODE, language="python")
