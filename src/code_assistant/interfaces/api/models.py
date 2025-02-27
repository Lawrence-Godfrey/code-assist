from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime
from enum import Enum

from code_assistant.models.types import MessageRole, StageStatus


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    

class MessageResponse(BaseModel):
    id: int
    content: str
    stage_id: int
    role: MessageRole
    role_display: str
    timestamp: Optional[datetime] = None
    
    class Config:
        use_enum_values = True

class MessageCreate(BaseModel):
    content: str
    role: MessageRole

    class Config:
        use_enum_values = True


class PipelineRequest(BaseModel):
    prompt_model_name: Optional[str] = "gpt-4"
    message_history: List[MessageCreate]


class PipelineResponse(BaseModel):
    status: str
    response: MessageCreate
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


class StageCreate(BaseModel):
    name: str
    status: Optional[str] = None


class ChatCreate(BaseModel):
    description: Optional[str] = None
    create_default_stages: bool = True


class StageResponse(BaseModel):
    id: int
    name: str
    status: StageStatus
    created_at: datetime
    next_stage_id: Optional[int] = None

    class Config:
        use_enum_values = True


class ChatResponse(BaseModel):
    id: int
    created_at: datetime
    stages: List[StageResponse] = []