""" Simple Chat Example

Standalone demo page showing customers the plain chat call behind the
chatbot, with no vector search involved - just a question straight to the
model.
"""
import logging

import streamlit as st

from ai_gateway import AIGateway
from constants import AppUserInterfaceElements
from constants import SessionStateVariables
from chrome import setup_example_page

logger = logging.getLogger(__name__)

card = setup_example_page(AppUserInterfaceElements.SIMPLE_CHAT_PAGE_TITLE, "💬")

with card:
    st.title(AppUserInterfaceElements.SIMPLE_CHAT_PAGE_TITLE)
    st.write("This is the plain chat call behind the chatbot, with no "
             "vector search involved - just a question straight to the "
             "model.")

    EXAMPLE_CODE = '''from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8321/v1/",  # OGX endpoint
    api_key="your-api-key",
)

response = client.responses.create(
    model="vllm-inference/gemma-4",
    input="What's the proper tire pressure for a 1978 Corvette?",
)

print(response.output_text)
'''
    st.code(EXAMPLE_CODE, language="python")

    st.divider()

    if SessionStateVariables.GATEWAY not in st.session_state:
        logger.info("Initializing OpenAI Client for Simple Chat Example page")
        gateway = AIGateway()
        gateway.connect()
        st.session_state[SessionStateVariables.GATEWAY] = gateway
    gateway = st.session_state[SessionStateVariables.GATEWAY]

    st.caption(f"Connected to the same OGX endpoint as the chatbot: {gateway.openai_client.base_url}")

    with st.form("simple_chat_form"):
        message = st.text_input("Ask the model anything",
                                placeholder="e.g. What's the proper tire pressure for a 1978 Corvette?")
        submitted = st.form_submit_button("Send")

    if submitted and message:
        try:
            response = gateway.openai_client.responses.create(
                model=gateway.model,
                input=message,
            )
            st.write(response.output_text)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Simple chat failed: %s", e)
            st.error(f"Chat failed: {e}")
    elif submitted:
        st.warning("Enter a message first.")
