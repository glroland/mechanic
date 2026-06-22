""" Mechanic Chatbot

Digital expert in fixing old corvettes
"""
import os
import uuid
import logging
import base64
import mlflow
import streamlit as st
from constants import SessionStateVariables
from constants import AppUserInterfaceElements
from constants import CannedGreetings
from constants import MessageAttributes
from constants import EnvironmentVariables
from ai_gateway import AIGateway

logger = logging.getLogger(__name__)

log_level = logging.INFO
if EnvironmentVariables.LOG_LEVEL in os.environ:
    if os.environ[EnvironmentVariables.LOG_LEVEL] is not None and len(os.environ[EnvironmentVariables.LOG_LEVEL]) > 0:
        log_level = os.environ[EnvironmentVariables.LOG_LEVEL]
logging.basicConfig(level=log_level,
    handlers=[
        # no need from a docker container - logging.FileHandler("mechanic-chatbot.log"),
        logging.StreamHandler()
    ])
logger.info("Logging initialized.")
mlflow_logger = logger = logging.getLogger("mlflow")
logger.setLevel(log_level)

# Prepare the MLFlow security token
if EnvironmentVariables.MLFLOW_TRACKING_TOKEN not in os.environ or \
        os.environ[EnvironmentVariables.MLFLOW_TRACKING_TOKEN] is None or \
        len(os.environ[EnvironmentVariables.MLFLOW_TRACKING_TOKEN]) == 0:
    logger.warning("MLFlow Token Not Set...")

    token_file_path = None
    if EnvironmentVariables.SET_TOKEN_FROM_FILE in os.environ:
        token_file_path = os.environ[EnvironmentVariables.SET_TOKEN_FROM_FILE]
    if token_file_path is not None and len(token_file_path) > 0:
        if os.path.exists(token_file_path):
            with open(token_file_path, "r") as f:
                # Strip whitespace/newlines to avoid HTTP header parsing bugs
                os.environ[EnvironmentVariables.MLFLOW_TRACKING_TOKEN] = f.read().strip()
        else:
            logger.error("MLFlow Token File is set but the file does not exist!  Filename = %s", token_file_path)
    else:
        logger.warning("MLFlow Token File Path Not Set Either!")

# Prepare engine bay photo
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
bin_str = get_base64_of_bin_file('assets/engine_bay.jpeg')

# Initialize Streamlit State
if SessionStateVariables.MESSAGES not in st.session_state:
    logger.info("Initializing OpenAI Client")
    gateway = AIGateway()
    gateway.connect()
    st.session_state[SessionStateVariables.GATEWAY] = gateway
    logger.info("Client Initialized")

    logger.info("Clearing message history.")
    st.session_state[SessionStateVariables.MESSAGES] = []

    experiment_name = os.environ.get(AIGateway.ENV_MLFLOW_EXPERIMENT_NAME, AIGateway.DEFAULT_EXPERIMENT_NAME)
    mlflow.set_experiment(experiment_name)
    session_tag = uuid.uuid4().hex[:8]
    with mlflow.start_run(run_name=f"chat-{session_tag}") as run:
        st.session_state[SessionStateVariables.MLFLOW_RUN_ID] = run.info.run_id
    logger.info("MLflow session run created: %s", st.session_state[SessionStateVariables.MLFLOW_RUN_ID])

# Rehydrate global variables from session state
gateway = st.session_state[SessionStateVariables.GATEWAY]

# Initialize High Level Page Structure
st.set_page_config(page_title=AppUserInterfaceElements.TITLE,
                   page_icon=AppUserInterfaceElements.TAB_ICON,
                   layout="wide")

# Page setup
css = f"""
<style>
.stMain {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    }}

.st-key-chatbot {{
    background-color: black;
    //background-image: linear-gradient(to right, rgba(255,255,255, 0.3) 0 100%), url("data:image/png;base64,{bin_str}");
    background-size: cover;
    background-attachment: local;
    //background-position: center center;
}}

.stAppHeader {{visibility: hidden;}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

@st.dialog("Service Manual", width="large")
def display_service_manual():
    st.pdf("assets/c3_repair.pdf")

# Setup header
col1, col2, col3 = st.columns([0.1, 0.2, 0.7], vertical_alignment="center")
with col1:
    st.image("assets/side_view_car.jpeg", width=100)
with col2:
    if st.button("Service Manual"):
        display_service_manual()
with col3:
    st.markdown(""" # My&nbsp;Classic&nbsp;Corvette&nbsp;Garage """)

# Initialize Chat Box
messages = st.container(height=400, key="chatbot")
messages.chat_message(MessageAttributes.ASSISTANT).write(CannedGreetings.INTRO + CannedGreetings.BREAK + f"I am powered by '{gateway.get_pretty_model_name()}'." + CannedGreetings.BREAK + CannedGreetings.ASK)
for msg in st.session_state.messages:
    messages.chat_message(msg[MessageAttributes.ROLE]).write(msg[MessageAttributes.CONTENT])

# Gather and log user prompt
if user_input := st.chat_input():
    logger.info ("User Input: %s", user_input)
    messages.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    logger.info ("st.session_state.messages - %s", st.session_state.messages)

    # Search VDB for relevant content and process chat, grouped under the session run
    with mlflow.start_run(run_id=st.session_state[SessionStateVariables.MLFLOW_RUN_ID]):
        logger.debug("Searching Vector Store for enriched content...")
        matching_content = gateway.rag_search(user_input)
        expanded_user_input = ""
        if matching_content is not None and len(matching_content) > 0:
            expanded_user_input += "Context:\n"
            for c in matching_content:
                expanded_user_input += c + "\n"
            expanded_user_input += "\nQuestion:\n"
        expanded_user_input += user_input
        logger.info("Expanded User Prompt: %s", expanded_user_input)

        # Process chat
        ai_response = None
        with messages.chat_message(MessageAttributes.ASSISTANT):
            placeholder = st.empty()
            ai_response = gateway.process_user_chat(expanded_user_input, placeholder)
    logger.info ("AI Response Message: %s", ai_response)

    # Append AI Response to history
    st.session_state.messages.append(
        {
            MessageAttributes.ROLE: MessageAttributes.ASSISTANT,
            MessageAttributes.CONTENT: ai_response
        }
    )
