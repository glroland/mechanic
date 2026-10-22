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

logger = logging.getLogger(__name__)

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
    max_num_results=3,
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
                max_num_results=AIGateway.MAX_SEARCH_RESULTS,
                timeout=AIGateway.SEARCH_TIMEOUT,
            )

            rows = []
            for rank, result in enumerate(response.data, start=1):
                content = "\n\n".join(chunk.text for chunk in result.content)
                rows.append({"Rank": rank, "Score": round(result.score, 4), "Content": content})

            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No matching content found.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("RAG search failed: %s", e)
            st.error(f"Search failed: {e}")
    elif submitted:
        st.warning("Enter a search query.")
