from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from code_assistant.models.types import MessageRole, StageStatus


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


class StageCreate(BaseModel):
    name: str
    status: Optional[StageStatus] = None
    next_stage_id: Optional[int] = None
    pipeline_endpoint: Optional[str] = None

    class Config:
        use_enum_values = True


class StageUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[StageStatus] = None
    next_stage_id: Optional[int] = None
    pipeline_endpoint: Optional[str] = None
    
    def model_dump_updates(self) -> dict:
        """
        Override the model_dump method to only include fields that were explicitly set.
        This prevents None values from being passed to the database update.
        """
        data = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                data[field_name] = field_value

        return data

    class Config:
        use_enum_values = True


class ChatCreate(BaseModel):
    description: Optional[str] = None
    create_default_stages: bool = True


class StageResponse(BaseModel):
    id: int
    name: str
    status: StageStatus
    status_display: str
    created_at: datetime
    next_stage_id: Optional[int] = None
    pipeline_endpoint: Optional[str] = None

    class Config:
        use_enum_values = True


class ChatResponse(BaseModel):
    id: int
    created_at: datetime
    stages: List[StageResponse] = []
