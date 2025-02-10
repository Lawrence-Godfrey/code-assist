from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    
class ExtractRequest(BaseModel):
    repository_url: HttpUrl
    branch: Optional[str] = "main"

class ExtractResponse(BaseModel):
    repository_id: str
    status: str
    file_count: int
    
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