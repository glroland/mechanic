""" Responses API Vector Search Example

Standalone demo page showing customers how vector search can be wired
directly into the inference call via the Responses API's file_search tool,
instead of the chatbot's own two-step search-then-stuff-the-prompt flow.
See the commented-out tools=[...] call in ai_gateway.py's
process_user_chat() - this page is that path, enabled.
"""
import logging

import streamlit as st

from ai_gateway import AIGateway
from constants import AppUserInterfaceElements
from constants import SessionStateVariables
from chrome import setup_example_page

logger = logging.getLogger(__name__)

card = setup_example_page(AppUserInterfaceElements.INTEGRATED_SEARCH_PAGE_TITLE, "🧩")

with card:
    st.title(AppUserInterfaceElements.INTEGRATED_SEARCH_PAGE_TITLE)
    st.write("The chatbot searches the vector store itself, then stuffs the "
             "results into the prompt by hand. The Responses API can do "
             "this in a single call instead, via the file_search tool - "
             "the model decides when to search, and retrieval happens as "
             "part of inference.")

    EXAMPLE_CODE = '''from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8321/v1/",  # OGX endpoint
    api_key="your-api-key",
)

response = client.responses.create(
    model="vllm-inference/gemma-4",
    input="How do I check the wheel alignment?",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
    include=["file_search_call.results"],
)

print(response.output_text)
'''
    st.code(EXAMPLE_CODE, language="python")

    st.divider()

    if SessionStateVariables.GATEWAY not in st.session_state:
        logger.info("Initializing OpenAI Client for Integrated Vector Search Example page")
        gateway = AIGateway()
        gateway.connect()
        st.session_state[SessionStateVariables.GATEWAY] = gateway
    gateway = st.session_state[SessionStateVariables.GATEWAY]

    st.caption(f"Connected to the same OGX endpoint as the chatbot: {gateway.openai_client.base_url}")

    with st.form("integrated_search_form"):
        question = st.text_input("Ask a question about the service manual",
                                 placeholder="e.g. How do I check the wheel alignment?")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        try:
            vector_store_id = gateway.get_vector_store_id()
            response = gateway.openai_client.responses.create(
                model=gateway.model,
                input=question,
                tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
                include=["file_search_call.results"],
            )

            st.write(response.output_text)

            rows = []
            for item in response.output:
                if getattr(item, "type", None) != "file_search_call":
                    continue
                for rank, result in enumerate(getattr(item, "results", None) or [], start=1):
                    rows.append({
                        "Rank": rank,
                        "Score": getattr(result, "score", None),
                        "Content": getattr(result, "text", None) or getattr(result, "filename", ""),
                    })

            if rows:
                st.divider()
                st.caption("Chunks the model's file_search call retrieved:")
                st.dataframe(rows, use_container_width=True, hide_index=True)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Integrated vector search failed: %s", e)
            st.error(f"Ask failed: {e}")
    elif submitted:
        st.warning("Enter a question first.")
