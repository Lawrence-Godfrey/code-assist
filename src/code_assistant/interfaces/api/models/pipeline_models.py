from pydantic import BaseModel
from typing import List, Optional

from code_assistant.models.types import MessageRole


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ChatMessage(BaseModel):
    content: str
    role: MessageRole

    class Config:
        use_enum_values = True


class PipelineRequest(BaseModel):
    prompt_model_name: Optional[str] = "gpt-4"
    message_history: List[ChatMessage]


class PipelineResponse(BaseModel):
    status: str
    response: ChatMessage
    approval_needed: bool = False


class PromptRequest(BaseModel):
    codebase_name: str
    prompt: str
    model_id: Optional[str] = "gpt-4"

    class Config:
        protected_namespaces = ()


class PromptResponse(BaseModel):
    response: str
    context_files: List[str]
    model_used: str 

    class Config:
        protected_namespaces = () 