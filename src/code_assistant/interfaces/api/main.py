import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from code_assistant.storage import messages
from code_assistant.logging.logger import get_logger
from fastapi.responses import JSONResponse

logger = get_logger(__name__)


from code_assistant.interfaces.api.models import (
    ExtractRequest,
    ExtractResponse,
    PromptRequest,
    PromptResponse,
    ModelInfo
)

app = FastAPI(
    title="Code Assistant API",
    description="RESTful API for code extraction and analysis",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log incoming requests (raw JSON)
@app.middleware("http")
async def log_request(request: Request, call_next):
    body = await request.body()
    body_str = body.decode('utf-8') if body else "EMPTY"
    logger.info(f"Incoming request {request.method} {request.url} with headers: {dict(request.headers)} and body: {body_str}")
    
    # Reconstruct the request stream so that downstream handlers can read it
    async def receive():
        return {"type": "http.request", "body": body}
    request._receive = receive

    response = await call_next(request)
    return response

class StageMessage(BaseModel):
    content: str
    role: str

class StageCreate(BaseModel):
    name: str
    status: Optional[str] = "pending"

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Return list of available AI models."""
    # This could be expanded to dynamically check available models
    return [
        ModelInfo(
            id="gpt-4",
            name="GPT-4",
            description="Latest GPT-4 model from OpenAI"
        )
    ]

@app.post("/extract", response_model=ExtractResponse)
async def extract_code(request: ExtractRequest):
    """Extract code from a GitHub repository."""
    # Implementation would use your existing extraction logic
    # Return repository data and status
    pass

@app.post("/prompt/codebase", response_model=PromptResponse)
async def process_prompt(request: PromptRequest):
    """Process a prompt against a specific codebase using RAG."""
    # Implementation would use your RAG module
    # Return the AI response
    pass

@app.websocket("/ws/{stage_id}")
async def websocket_endpoint(websocket: WebSocket, stage_id: int):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time updates here
    except WebSocketDisconnect:
        pass

@app.post("/api/stages/{stage_id}/messages")
async def create_message(stage_id: int, message: StageMessage):
    saved_message = messages.save_message(stage_id, message.dict())
    return saved_message

@app.get("/api/stages/{stage_id}/messages")
async def get_messages(stage_id: int):
    stage_messages = messages.get_messages(stage_id)
    return stage_messages

@app.patch("/api/stages/{stage_id}")
async def update_stage(stage_id: int, update: dict):
    messages.update_pipeline_state(stage_id, update)
    new_state = messages.get_pipeline_state(stage_id)
    return {"status": "success", "pipeline_state": new_state}

@app.post("/api/stages")
async def create_stage(stage: StageCreate):
    new_stage = messages.create_stage(stage.dict())
    return new_stage

@app.get("/api/stages")
async def list_stages():
    stages = messages.get_stages()
    return stages

@app.get("/api/stages/{stage_id}")
async def get_stage(stage_id: int):
    stage = messages.get_stage(stage_id)
    if not stage:
        raise HTTPException(status_code=404, detail="Stage not found")
    return stage

# Add a global exception handler to abstract exception handling for all endpoints
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":

    load_dotenv()
    
    if os.environ.get("ENV") == "development":
        print("Running in development mode")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
    else:
        print("Running in production mode")
        uvicorn.run(app, host="0.0.0.0", port=8000) 