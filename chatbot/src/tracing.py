""" MLflow GenAI tracing setup for the chatbot.

Mirrors the tracing.py pattern used in the distribution-center project:
MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE, MLFLOW_TRACKING_AUTH, and friends are
read natively by mlflow from the environment -- nothing to configure here
beyond deciding whether tracing should run at all. Without a tracking URI
(e.g. running locally without `make run-chatbot`), tracing is disabled
outright so nothing is written to a local ./mlruns directory.

mlflow.openai.autolog() is safe to call unconditionally: it patches the
OpenAI client to emit a span per call, but those spans are no-ops while
tracing is disabled.
"""
import logging
import os

import mlflow

logger = logging.getLogger(__name__)

ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
DEFAULT_EXPERIMENT_NAME = "mechanic-chatbot-v2"


def configure_tracing() -> None:
    """ Enable MLflow tracing for the chatbot's LLM calls and vector search. """
    mlflow.openai.autolog()
    if not os.environ.get(ENV_MLFLOW_TRACKING_URI):
        mlflow.tracing.disable()
        return
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    logger.info("MLflow tracing enabled: uri=%s", os.environ[ENV_MLFLOW_TRACKING_URI])
