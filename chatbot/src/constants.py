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

# pylint: disable=too-few-public-methods
class CannedGreetings:
    """ Preestablished Responses """

    INTRO = "Welcome to the garage!!!  What are you working on today?"

# pylint: disable=too-few-public-methods
class MessageAttributes:
    """ LLM APU Message Attributes """

    ROLE = "role"
    USER = "user"
    ASSISTANT = "assistant"
    CONTENT = "content"
