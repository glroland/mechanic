""" Constants for the Baseball Chatbot Application
"""

AGENT_SYSTEM_PROMPT = """
    You are a classic car mechanic that specializes in C3/third-generation Corvettes.  

    Only use the context provided when answering technical questions about the vehicle and its repair or maintenance.
    
    Provide relatively concise responses where possible.
                      """

# pylint: disable=too-few-public-methods
class SessionStateVariables:
    """ Session State Variable Names """

    MESSAGES = "messages"
    MODEL = "model"
    GATEWAY = "gateway"
    RESPONSES = "responses"

# pylint: disable=too-few-public-methods
class AppUserInterfaceElements:
    """ Application UI Elements """

    TITLE = "Classic Corvette Chatbot"

    HEADER = "Classic&nbsp;Corvette&nbsp;Mechanic"

    TAB_ICON = "./assets/icon.png"

    RAG_SEARCH_PAGE_TITLE = "RAG Search Example"

    RETURN_TO_APP_NAV_LABEL = "Return to App"

    SIMPLE_CHAT_PAGE_TITLE = "Simple Chat Example"

    SIMPLE_CHAT_NAV_LABEL = "Simple Chat"

    RAG_SEARCH_NAV_LABEL = "RAG Search"

    INTEGRATED_SEARCH_PAGE_TITLE = "Responses API Vector Search Example"

    INTEGRATED_SEARCH_NAV_LABEL = "Integrated Vector Search"

    ADD_CONTENT_PAGE_TITLE = "Add Content Example"

    ADD_CONTENT_NAV_LABEL = "Add Content"

    CONVERT_DOCUMENT_PAGE_TITLE = "Convert Document Example"

    CONVERT_DOCUMENT_NAV_LABEL = "Convert Document"

# pylint: disable=too-few-public-methods
class CannedGreetings:
    """ Preestablished Responses """

    INTRO = "Welcome to the garage!!!"
    ASK = "How can I help you with your vette?"
    BREAK = "\n\n"

# pylint: disable=too-few-public-methods
class MessageAttributes:
    """ LLM APU Message Attributes """

    ROLE = "role"
    USER = "user"
    ASSISTANT = "assistant"
    CONTENT = "content"

# pylint: disable=too-few-public-methods
class EnvironmentVariables:
    """ Environment Variables """

    LOG_LEVEL = "LOG_LEVEL"
