# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


class SessionConfig(BaseModel):
    """Session formatting configuration for OpenViking."""

    speaker_labels: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Display labels to use when formatted session messages lack peer_id. "
            "Keys are message roles such as 'user' and 'assistant'. When unset, "
            "formatting falls back to the raw role string."
        ),
    )

    model_config = {"extra": "forbid"}

    @field_validator("speaker_labels", mode="before")
    @classmethod
    def normalize_speaker_labels(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("session.speaker_labels must be an object")

        labels: Dict[str, str] = {}
        for raw_role, raw_label in value.items():
            role = str(raw_role).strip()
            label = str(raw_label).strip()
            if role and label:
                labels[role] = label
        return labels

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SessionConfig":
        """Create configuration from dictionary."""
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
