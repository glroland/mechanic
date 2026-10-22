""" Add Content Example

Standalone demo page showing customers how new content gets added to the
vector store the chatbot searches, using the same upload-a-file-then-attach-
it-to-the-vector-store call ingest/src/import.py makes per chunk.
"""
import logging
from io import BytesIO

import streamlit as st

from ai_gateway import AIGateway
from constants import AppUserInterfaceElements
from constants import SessionStateVariables
from chrome import setup_example_page

logger = logging.getLogger(__name__)

card = setup_example_page(AppUserInterfaceElements.ADD_CONTENT_PAGE_TITLE, "📥")

with card:
    st.title(AppUserInterfaceElements.ADD_CONTENT_PAGE_TITLE)
    st.write("This is the same call the ingestion pipeline makes to add a new "
             "chunk of content to the vector store the chatbot searches.")
    st.warning("This adds real content to the live vector store used by the "
               "chatbot - only submit content you're OK having it answer from.")

    EXAMPLE_CODE = '''from io import BytesIO
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8321/v1/",  # OGX endpoint
    api_key="your-api-key",
)

content = "Torque the lug nuts to 80 ft-lbs in a star pattern."
uploaded_file = client.files.create(
    file=("note.txt", BytesIO(content.encode("utf-8")), "text/plain"),
    purpose="assistants",
)

client.vector_stores.files.create(
    vector_store_id=vector_store_id,
    file_id=uploaded_file.id,
)
'''
    st.code(EXAMPLE_CODE, language="python")

    st.divider()

    if SessionStateVariables.GATEWAY not in st.session_state:
        logger.info("Initializing OpenAI Client for Add Content Example page")
        gateway = AIGateway()
        gateway.connect()
        st.session_state[SessionStateVariables.GATEWAY] = gateway
    gateway = st.session_state[SessionStateVariables.GATEWAY]

    st.caption(f"Connected to the same OGX endpoint as the chatbot: {gateway.openai_client.base_url}")

    with st.form("add_content_form"):
        content = st.text_area("Content to add",
                               placeholder="e.g. Torque the lug nuts to 80 ft-lbs in a star pattern.")
        submitted = st.form_submit_button("Add to Vector Store")

    if submitted and content:
        try:
            vector_store_id = gateway.get_vector_store_id()
            uploaded_file = gateway.openai_client.files.create(
                file=("web-submission.txt", BytesIO(content.encode("utf-8")), "text/plain"),
                purpose="assistants",
            )
            vector_store_file = gateway.openai_client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=uploaded_file.id,
            )

            st.success("Content added to the vector store.")
            st.dataframe([{
                "File ID": uploaded_file.id,
                "Vector Store File ID": vector_store_file.id,
                "Status": vector_store_file.status,
            }], use_container_width=True, hide_index=True)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Add content failed: %s", e)
            st.error(f"Add content failed: {e}")
    elif submitted:
        st.warning("Enter some content to add.")
