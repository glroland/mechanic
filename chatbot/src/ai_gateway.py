"""API Utilities

Utility functions related to connecting with the backend services supporting
the chatbot.
"""
import os
import logging
import streamlit as st
from constants import AGENT_SYSTEM_PROMPT

import mlflow
from mlflow.entities import SpanType
mlflow.openai.autolog()
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIGateway:

    ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
    ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
    ENV_MODEL = "MODEL"

    ENV_MLFLOW_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
    DEFAULT_EXPERIMENT_NAME = "mechanic.chatbot"

    MECHANIC_VECTOR_DB_NAME = "mechanic_vector_db"

    SEARCH_TIMEOUT = 5 * 60
    MAX_SEARCH_RESULTS = 3

    openai_client : OpenAI = None
    previous_response_id = None
    model = None
    vector_store_id = None

    def connect(self):
        """ Connects to the remote service provider. """
        # Configure MLflow tracing
        experiment_name = os.environ.get(
            self.ENV_MLFLOW_EXPERIMENT_NAME,
            self.DEFAULT_EXPERIMENT_NAME
        )
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow experiment: %s", experiment_name)
        mlflow.openai.autolog()
        logger.info("MLflow OpenAI autologging enabled.")

        # get the base url
        if not self.ENV_OPENAI_BASE_URL in os.environ:
            msg = "OpenAI Base URL has not been set and is a required variable. 'OPENAI_BASE_URL' missing."
            logger.error(msg)
            raise ValueError(msg)
        openai_base_url = os.environ[self.ENV_OPENAI_BASE_URL]
        logger.info("OpenAI Base URL: %s", openai_base_url)

        # get the api key
        if not self.ENV_OPENAI_API_KEY in os.environ:
            msg = "OpenAI API Key is a required environment variable.  'OPENAI_API_KEY' missing."
            logger.error(msg)
            raise ValueError(msg)
        api_key = os.environ[self.ENV_OPENAI_API_KEY]
        logger.info("OpenAI API Key: %s", api_key)

        # Connect to OpenaI
        logger.info("Connecting to OpenAI.  URL=%s", openai_base_url)
        self.openai_client = OpenAI(base_url = openai_base_url,
                               api_key = api_key)
        logger.info("Successfully connected to OpenAI.")

        # get configured model
        if not self.ENV_MODEL in os.environ:
            msg = "OpenAI Model is a required environment variable.  'MODEL' missing."
            logger.error(msg)
            raise ValueError(msg)
        self.model = os.environ[self.ENV_MODEL]
        if len(self.model) == 0:
            msg = "OpenAI Model is empty and required."
            logger.error(msg)
            raise ValueError(msg)
        logger.info("OpenAI Model: %s", self.model)

    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def process_user_chat(self, user_input: str, placeholder) -> str:
        """ Process a chat request.
        
            user_input - user message
            placeholder - streamlit placeholder
            
            Returns: Chat response
        """
        # store inputs
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_inputs({"user_input": user_input})

        # Log the user chat and systems prompt
        logger.info("System Prompt: %s", AGENT_SYSTEM_PROMPT)
        logger.info("User Input: %s", user_input)

        # get vector store id
        #vs_id = self.get_vector_store_id()

        # Employ OpenAI Responses AI
        response_stream = self.openai_client.responses.create(
            model=self.model,
            instructions=AGENT_SYSTEM_PROMPT,
            input=user_input,
            temperature=0.3,
            store=True,
            previous_response_id=self.previous_response_id,
            stream=True,
            #tools=[{"type": "file_search", "vector_store_ids": [vs_id]}],
            #include=["file_search_call.results"]
        )

        # Capture response
        ai_response = ""
        for event in response_stream:
            if hasattr(event, "type") and "text.delta" in event.type:
                ai_response += event.delta
                print(event.delta, end="", flush=True)
                with placeholder.container():
                    st.write(ai_response)
            elif hasattr(event, "type") and "response.completed" in event.type:
                self.previous_response_id = event.response.id

        return ai_response

    def get_vector_store_id(self):
        """ Gets the vector store Id for the application.

            Returns: vector store id
        """
        # use cache
        if self.vector_store_id is not None:
            return self.vector_store_id

        # find vector store id
        vector_stores = self.openai_client.vector_stores.list()
        logger.info("Vector Stores: %s", vector_stores)
        for vs in vector_stores:
            if vs.name == self.MECHANIC_VECTOR_DB_NAME:
                logger.debug("Vector Store ID: %s", vs.id)
                self.vector_store_id = vs.id
                return self.vector_store_id

        # no vector database - fail
        msg = "Vector Database must be created and populated before running web application!"
        logger.error(msg)
        raise ValueError(msg)

    def get_pretty_model_name(self):
        """ Gets the beautified name of the enabled model.
        
            Returns: model name
        """
        parts = os.path.split(self.model)
        return parts[len(parts) - 1]

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def rag_search(self, query:str, max_matches:int=None):
        """ Searches the vector store for a match to the provided user query. 
        
            query - user query
        """
        # store inputs
        span = mlflow.get_current_active_span()
        if span is not None:
            span.set_inputs({"query": query, "max_matches": max_matches})

        # get vector store
        vector_store_id = self.get_vector_store_id()

        # determine max results
        max = self.MAX_SEARCH_RESULTS
        if max_matches is not None:
            max = max_matches
        logger.debug("Max Matches: %s", max)

        # Perform vector search
        vs_response = self.openai_client.vector_stores.search(
            vector_store_id=vector_store_id,
            query=query,
            max_num_results=max,
            timeout=self.SEARCH_TIMEOUT
            )

        # extract meaningful content from results
        results = []
        logger.debug("VDB Search Response: %s", vs_response)
        for r in vs_response.data:
            matching_set = []
            for c in r.content:
                matching_set.append(c.text)
            logger.info("===== MATCHING CONTENT =====  Score=%s. Content=%s", r.score, matching_set)
            results = results + matching_set

        logger.debug("Vector Search Results: %s", results)
        return results
