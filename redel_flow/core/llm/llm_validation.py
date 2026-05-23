"""Validation models for structured LLM outputs used by the agent framework."""

import json
import logging
from typing import Dict, Union

from json_repair import repair_json
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

def try_json_loads(s: str):
    try:
        return json.loads(s, strict=False)
    except json.JSONDecodeError as e:
        logger.error("[messages] json.loads failed: %s | sample=%r", e, s)
        return None


class CallAgentsWithMessages(BaseModel):
    """Schema used when the agent needs to contact one or more other agents with specific tasks or questions."""

    @model_validator(mode="after")
    def messages_not_empty(self):
        if not self.messages:
            raise ValueError(
                "EMPTY_MESSAGES"
            )
        return self

    @field_validator("messages", mode="before")
    @classmethod
    def parse_messages(cls, v):

        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            parsed = try_json_loads(v)
            if isinstance(parsed, dict):
                return parsed

        repaired = repair_json(v)
        if repaired == "":
            raise ValueError(
                "INVALID_MESSAGES"
            )

        if isinstance(repaired, dict):
            return repaired

        if isinstance(repaired, str):
            parsed = try_json_loads(repaired)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError("INVALID_MESSAGES")

    messages: Dict[str, str] = Field(
        #description="A dictionary mapping agent names to the messages you want to send them. Each key is the agent's name and the value is the specific message for that agent. Example: {'Agent1': 'Message for Agent 1', 'Agent2': 'Message for Agent 2'}."
        description="JSON object mapping agent names to their message. Escape double quotes (\\\"). Example: {\"Agent1\":\"Message for Agent 1.\", \"Agent2\":\"Message with \\\"quoted\\\" text.\"}"
    )
    current_state: str = Field(
        description="Explain your reasoning and motivations for contacting these agents. This should summarize the current understanding of the task and what information you seek."
    )


class SendResultToCaller(BaseModel):
    """Schema used when the agent has completed the task and is ready to send a final answer to the requester."""

    result: str = Field(
        description="The final message or conclusion that should be sent back to the original caller. It should summarize the outcome of the agent's reasoning or delegated actions."
    )
    current_state: str = Field(
        description="Summarize the final reasoning that led to this result. Mention any tools used, information retrieved, or agents contacted."
    )



