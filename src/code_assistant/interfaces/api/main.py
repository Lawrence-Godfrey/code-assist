import os
from typing import List
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from code_assistant.models.factory import ModelFactory
from code_assistant.models.prompt import PromptModel
from code_assistant.models.types import Message as MessageModel
from code_assistant.storage.chats import Chat, Message, Stage, get_db
from code_assistant.logging.logger import get_logger
from code_assistant.interfaces.api.models import (
    ChatCreate,
    MessageCreate,
    PipelineRequest,
    PipelineResponse,
    ModelInfo,
    MessageResponse,
    StageCreate,
    MessageRole,
    ChatResponse,
    StageResponse
)


logger = get_logger(__name__)


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

@app.post("/api/pipeline", response_model=PipelineResponse)
async def pipeline(request: PipelineRequest):
    """Execute a pipeline of steps."""
    prompt_model: PromptModel = ModelFactory.create(request.prompt_model_name)
    
    # Check if this is an initial prompt by looking for a system message
    has_system_message = any(msg.role == MessageRole.SYSTEM for msg in request.message_history)

    message_history = [MessageModel(content=msg.content, role=msg.role) for msg in request.message_history]

    if not has_system_message:
        # If no system message exists, this is the initial prompt
        system_message = MessageModel(
            content="""You are a requirements analysis assistant. 
            Your task is to analyze business requirements and request feedback if needed. 
            Once you have the necessary information, you will generate a summary of the requirements, 
            and ask the user to approve the summary in order to proceed. This final response should contain the keyword 'APPROVAL_NEEDED' explicitly.
            The users who will be interacting with you are business users who are not technical and will not be able to provide you with the exact requirements.
            """,
            role=MessageRole.SYSTEM,
        )
        messages_for_model = [system_message, *message_history]
    else:
        # System message already exists, use chat history as is
        messages_for_model = message_history
    
    # Generate response using the chat history
    response_content = prompt_model.generate_response(
        messages=messages_for_model,
        temperature=0.1,  # Low temperature for more consistent analysis
    )

    # Create the assistant's response message
    assistant_response_message = MessageCreate(
        content=response_content,
        role=MessageRole.ASSISTANT
    )

    return PipelineResponse(
        status="success",
        response=assistant_response_message,
        approval_needed="APPROVAL_NEEDED" in response_content
    )


@app.websocket("/ws/{stage_id}")
async def websocket_endpoint(websocket: WebSocket, stage_id: int):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time updates here
    except WebSocketDisconnect:
        pass


@app.post("/api/chats", response_model=ChatResponse)
def create_chat(request: ChatCreate):
    with get_db() as session:
        request_dict = request.model_dump()
        create_default_stages = request_dict.pop("create_default_stages")
        chat = Chat.create(session, **request_dict)

        if create_default_stages:
            new_stages = [StageResponse(
                id=stage.id,
                name=stage.name,
                status=stage.status,
                created_at=stage.created_at,
                next_stage_id=stage.next_stage_id
            ) for stage in Stage.create_default_stages(session, chat.id)]
        else:
            new_stages = []

        return ChatResponse(
            id=chat.id,
            created_at=chat.created_at,
            stages=new_stages
        )

@app.get("/api/chats", response_model=List[ChatResponse])
def list_chats():
    with get_db() as session:
        chats = Chat.get_all(session)
        logger.info(f"Retrieved {len(chats)} chats: {chats}")
        print(f"Chats: {chats}")
        return [
            ChatResponse(
                id=chat.id,
                created_at=chat.created_at
            )
            for chat in chats
        ]
        
@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int):
    with get_db() as session:
        # First delete all messages in all stages of this chat
        stages = Stage.get_all(session, chat_id=chat_id)
        for stage in stages:
            messages = Message.get_all(session, stage_id=stage.id)
            for message in messages:
                Message.delete(session, message.id)
                
        # Next delete all stages
        for stage in stages:
            Stage.delete(session, stage.id)
            
        # Finally delete the chat
        success = Chat.delete(session, chat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
            
        return {"success": True}


@app.post("/api/stages/{stage_id}/messages", response_model=MessageResponse)
async def create_message(stage_id: int, message: MessageCreate):
    with get_db() as session:
        saved_message = Message.create(session, stage_id=stage_id, **message.model_dump())
        return MessageResponse(
            id=saved_message.id,
            stage_id=saved_message.stage_id,
            content=saved_message.content,
            role=saved_message.role,
            role_display=saved_message.get_role_display(),
            timestamp=saved_message.timestamp
        )


@app.get("/api/stages/{stage_id}/messages", response_model=List[MessageResponse])
async def get_messages(stage_id: int):
    with get_db() as session:
        messages = Message.get_all(session, stage_id=stage_id)
        return [
            MessageResponse(
                id=message.id,
                stage_id=message.stage_id,
                content=message.content,
                role=message.role,
                role_display=message.get_role_display(),
                timestamp=message.timestamp
            )
            for message in messages
        ]


@app.patch("/api/stages/{stage_id}")
async def update_stage(stage_id: int, **kwargs):
    with get_db() as session:
        updated_stage = Stage.update(session, stage_id, **kwargs)
        return StageResponse(
            id=updated_stage.id,
            name=updated_stage.name,
            status=updated_stage.status,
            created_at=updated_stage.created_at,
            next_stage_id=updated_stage.next_stage_id
        )

@app.post("/api/chats/{chat_id}/stages", response_model=StageResponse)
async def create_stage(chat_id: int, stage: StageCreate):
    with get_db() as session:
        new_stage = Stage.create(session, chat_id=chat_id, **stage.model_dump())
        logger.info(f"Created new stage: {new_stage}")
        return StageResponse(
            id=new_stage.id,
            name=new_stage.name,
            status=new_stage.status,
            created_at=new_stage.created_at,
            next_stage_id=new_stage.next_stage_id
        )

@app.get("/api/chats/{chat_id}/stages", response_model=List[StageResponse])
async def list_stages(chat_id: int):
    with get_db() as session:
        stages = Stage.get_all(session, chat_id=chat_id)
        return [
            StageResponse(
                id=stage.id,
                name=stage.name,
                status=stage.status,
                created_at=stage.created_at,
                next_stage_id=stage.next_stage_id
            )
            for stage in stages
        ]

@app.get("/api/stages/{stage_id}")
async def get_stage(stage_id: int):
    with get_db() as session:
        stage = Stage.get_by_id(session, stage_id)
        if not stage:
            raise HTTPException(status_code=404, detail="Stage not found")
        return StageResponse(
            id=stage.id,
            name=stage.name,
            status=stage.status,
            created_at=stage.created_at,
            next_stage_id=stage.next_stage_id
        )

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