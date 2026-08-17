"""Public contracts for the Text2Model Forge asset compiler."""

__version__ = "0.2.0rc1"

from .artifacts import ArtifactStore
from .compiler import Text2ModelCompiler
from .events import EventLog
from .lineage import evaluate_release, evaluate_research
from .localdeploy import LocalDeployStructuredClient, StructuredResponseError
from .operations import OperationRegistry
from .qualification import evaluate_legacy_component
from .registry import load_registry

__all__ = [
    "ArtifactStore",
    "Text2ModelCompiler",
    "EventLog",
    "LocalDeployStructuredClient",
    "OperationRegistry",
    "StructuredResponseError",
    "evaluate_legacy_component",
    "evaluate_release",
    "evaluate_research",
    "load_registry",
]
