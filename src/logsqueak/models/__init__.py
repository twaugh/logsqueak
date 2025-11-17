"""Pydantic data models for Logsqueak."""

# Import models in dependency order to resolve forward references
from logsqueak.models.edited_content import EditedContent
from logsqueak.models.integration_decision import IntegrationDecision

# Rebuild IntegrationDecision model now that EditedContent is defined
IntegrationDecision.model_rebuild()
