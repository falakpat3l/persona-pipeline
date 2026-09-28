"""persona-pipeline: AI orchestration for consistent virtual-persona content."""

from persona_pipeline.config import Config, load_config
from persona_pipeline.models import Brief, PostJob
from persona_pipeline.pipeline import Pipeline

__version__ = "0.1.0"

__all__ = ["Brief", "Config", "Pipeline", "PostJob", "load_config", "__version__"]
