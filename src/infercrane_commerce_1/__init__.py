"""Local serving surface for a content-identified InferCrane Commerce-1."""

from .runtime import (
    API_SCHEMA_VERSION,
    MODEL_ID,
    RELEASE_STATUS,
    DecisionRuntime,
    ModelIdentity,
    RuntimePrediction,
)

__all__ = [
    "API_SCHEMA_VERSION",
    "MODEL_ID",
    "RELEASE_STATUS",
    "DecisionRuntime",
    "ModelIdentity",
    "RuntimePrediction",
]

__version__ = "0.1.0"
