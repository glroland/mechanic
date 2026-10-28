""" RAG Search Example

Standalone demo page showing customers the vector search call the chatbot
makes to retrieve context chunks before answering, and letting them run it
themselves against the live vector store.
"""
import logging

import streamlit as st

from ai_gateway import AIGateway
from constants import AppUserInterfaceElements
from constants import SessionStateVariables
from chrome import setup_example_page
from chrome import render_results_table

logger = logging.getLogger(__name__)

# More than the chatbot's own AIGateway.MAX_SEARCH_RESULTS, so customers can
# see how relevance falls off further down the result list
MAX_RESULTS = 10

card = setup_example_page(AppUserInterfaceElements.RAG_SEARCH_PAGE_TITLE, "🔍")

with card:
    st.title(AppUserInterfaceElements.RAG_SEARCH_PAGE_TITLE)
    st.write("This is the same vector search call the chatbot makes against the "
             "service manual before answering a question, simplified so you can "
             "try it with your own queries below.")

    EXAMPLE_CODE = '''from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8321/v1/",  # OGX endpoint
    api_key="your-api-key",
)

response = client.vector_stores.search(
    vector_store_id=vector_store_id,
    query="How do I check the wheel alignment?",
    max_num_results=10,  # keep this small (e.g. 3-5) when feeding results to an LLM
)

for result in response.data:
    for chunk in result.content:
        print(f"Score: {result.score:.3f}  Text: {chunk.text}")
'''
    st.code(EXAMPLE_CODE, language="python")

    st.divider()

    if SessionStateVariables.GATEWAY not in st.session_state:
        logger.info("Initializing OpenAI Client for RAG Search Example page")
        gateway = AIGateway()
        gateway.connect()
        st.session_state[SessionStateVariables.GATEWAY] = gateway
    gateway = st.session_state[SessionStateVariables.GATEWAY]

    st.caption(f"Connected to the same OGX endpoint as the chatbot: {gateway.openai_client.base_url}")

    with st.form("rag_search_form"):
        query = st.text_input("Search the service manual",
                               placeholder="e.g. How do I check the wheel alignment?")
        submitted = st.form_submit_button("Search")

    if submitted and query:
        try:
            vector_store_id = gateway.get_vector_store_id()
            response = gateway.openai_client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=MAX_RESULTS,
                timeout=AIGateway.SEARCH_TIMEOUT,
            )

            results = [(result.score, "\n\n".join(chunk.text for chunk in result.content))
                       for result in response.data]

            if results:
                render_results_table(results)
            else:
                st.info("No matching content found.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("RAG search failed: %s", e)
            st.error(f"Search failed: {e}")
    elif submitted:
        st.warning("Enter a search query.")
