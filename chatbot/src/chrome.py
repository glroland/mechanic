""" Shared page chrome: the engine-bay background image and the examples
sidebar nav, used by every page (the main chatbot page and the example
pages) so both stay in sync instead of being redefined per file.

The sidebar is always present, just collapsed by default
(initial_sidebar_state="collapsed" in each page's st.set_page_config) -
that's a framework-level initial render setting, not CSS, so the sidebar
never has to "pop in" after the fact.
"""
import base64

import streamlit as st

from constants import AppUserInterfaceElements

_NAV_LINKS = [
    ("app.py", AppUserInterfaceElements.RETURN_TO_APP_NAV_LABEL, "↩️"),
    ("pages/simple_chat_example.py", AppUserInterfaceElements.SIMPLE_CHAT_NAV_LABEL, "💬"),
    ("pages/rag_search_example.py", AppUserInterfaceElements.RAG_SEARCH_NAV_LABEL, "🔍"),
    ("pages/integrated_vector_search_example.py", AppUserInterfaceElements.INTEGRATED_SEARCH_NAV_LABEL, "🧩"),
    ("pages/add_content_example.py", AppUserInterfaceElements.ADD_CONTENT_NAV_LABEL, "📥"),
    ("pages/convert_document_example.py", AppUserInterfaceElements.CONVERT_DOCUMENT_NAV_LABEL, "📄"),
]

EXAMPLE_CARD_CSS = """
<style>
.st-key-example_card {
    background-color: #ffffff;
    border-radius: 28px;
    padding: 2.5rem 3rem;
    max-width: 1100px;
    width: 92%;
    margin: 2rem auto;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.45);
}
.st-key-example_card, .st-key-example_card p, .st-key-example_card span,
.st-key-example_card label, .st-key-example_card h1, .st-key-example_card h2,
.st-key-example_card h3, .st-key-example_card div {
    color: #1a1a1a;
}
.st-key-example_card pre, .st-key-example_card code {
    background-color: #f5f5f5 !important;
    color: #1a1a1a !important;
    border-radius: 12px;
}
.st-key-example_card div[data-testid="stTextInput"] input,
.st-key-example_card div[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #cccccc !important;
    border-radius: 10px;
}
.st-key-example_card [data-testid="stDataFrame"] {
    background-color: #ffffff !important;
}
</style>
"""


def get_base64_of_bin_file(bin_file: str) -> str:
    """ Base64-encodes a binary file for embedding as a CSS data URI. """
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def apply_background() -> None:
    """ The engine-bay background image, active on every page, plus hiding
        Streamlit's auto-generated page list (replaced on every page by
        render_sidebar_nav()'s own links, so it doesn't show both).
    """
    bin_str = get_base64_of_bin_file('assets/engine_bay.jpeg')
    st.markdown(f"""
    <style>
    .stMain {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    [data-testid="stSidebarNav"] {{
        display: none;
    }}
    [data-testid="stExpandSidebarButton"] {{
        visibility: visible !important;
        opacity: 1 !important;
        background-color: rgba(0, 0, 0, 0.45) !important;
        border-radius: 10px !important;
        padding: 0.4rem !important;
    }}
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] span {{
        visibility: visible !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        font-size: 2.5rem !important;
        fill: #ffffff !important;
        color: #ffffff !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_nav() -> None:
    """ The persistent, collapsed-by-default examples nav. """
    with st.sidebar:
        for page, label, icon in _NAV_LINKS:
            st.page_link(page, label=label, icon=icon)


def setup_example_page(page_title: str, page_icon: str):
    """ Configures an example page and returns the white, rounded content
        card that floats over the shared background image - callers should
        put their content in a `with` block on the returned container.
    """
    st.set_page_config(page_title=page_title,
                       page_icon=page_icon,
                       layout="wide",
                       initial_sidebar_state="collapsed")
    apply_background()
    render_sidebar_nav()
    st.markdown(EXAMPLE_CARD_CSS, unsafe_allow_html=True)
    return st.container(key="example_card")
