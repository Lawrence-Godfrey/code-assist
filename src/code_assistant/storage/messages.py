import os
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Read the database URL from environment; default to SQLite for development
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# Create the engine; for SQLite, add connect_args to allow use in a multithreaded context
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    stage_id = Column(Integer, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PipelineState(Base):
    __tablename__ = "pipeline_states"
    stage_id = Column(Integer, primary_key=True, index=True)
    state = Column(JSON, nullable=False)

class Stage(Base):
    __tablename__ = "stages"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    next_stage_id = Column(Integer, nullable=True)  # Reference to the next stage in the pipeline


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


def save_message(stage_id: int, message: dict) -> None:
    session = SessionLocal()
    try:
        msg_instance = Message(
            stage_id=stage_id,
            role=message.get("role", ""),
            content=message.get("content", ""),
            timestamp=datetime.utcnow()
        )
        session.add(msg_instance)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_messages(stage_id: int) -> List[dict]:
    session = SessionLocal()
    try:
        msgs = session.query(Message).filter(Message.stage_id == stage_id).all()
        return [{
            "id": msg.id,
            "stage_id": msg.stage_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        } for msg in msgs]
    finally:
        session.close()


def update_pipeline_state(stage_id: int, updates: dict) -> None:
    session = SessionLocal()
    try:
        pipeline = session.query(PipelineState).filter(PipelineState.stage_id == stage_id).first()
        if pipeline is None:
            pipeline = PipelineState(stage_id=stage_id, state=updates)
            session.add(pipeline)
        else:
            if isinstance(pipeline.state, dict):
                pipeline.state.update(updates)
            else:
                pipeline.state = updates
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_pipeline_state(stage_id: int) -> dict:
    session = SessionLocal()
    try:
        pipeline = session.query(PipelineState).filter(PipelineState.stage_id == stage_id).first()
        if pipeline:
            return pipeline.state
        return {}
    finally:
        session.close()


def create_stage(stage_data: dict) -> dict:
    session = SessionLocal()
    try:
        stage = Stage(
            name=stage_data["name"],
            status=stage_data.get("status", "pending"),
            next_stage_id=stage_data.get("next_stage_id")
        )
        session.add(stage)
        session.commit()
        session.refresh(stage)
        return {
            "id": stage.id,
            "name": stage.name,
            "status": stage.status,
            "created_at": stage.created_at.isoformat(),
            "next_stage_id": stage.next_stage_id
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_stage(stage_id: int) -> dict:
    session = SessionLocal()
    try:
        stage = session.query(Stage).filter(Stage.id == stage_id).first()
        if stage:
            return {
                "id": stage.id,
                "name": stage.name,
                "status": stage.status,
                "created_at": stage.created_at.isoformat(),
                "next_stage_id": stage.next_stage_id
            }
        return {}
    finally:
        session.close()


def get_stages() -> List[dict]:
    session = SessionLocal()
    try:
        stages = session.query(Stage).all()
        return [{
            "id": stage.id,
            "name": stage.name,
            "status": stage.status,
            "created_at": stage.created_at.isoformat(),
            "next_stage_id": stage.next_stage_id
        } for stage in stages]
    finally:
        session.close() 