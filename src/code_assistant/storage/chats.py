import os
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from datetime import datetime, timezone
from contextlib import contextmanager
from enum import Enum as PyEnum

from code_assistant.interfaces.api.models.chat_models import MessageRole
from code_assistant.models.types import StageStatus


# Read the database URL from environment; default to SQLite for development
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

# Create the engine; for SQLite, add connect_args to allow use in a multithreaded context
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


class CRUDMixin:

    @classmethod
    def create(cls, session, **kwargs):
        instance = cls(**kwargs)
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance

    @classmethod
    def get_all(cls, session, **kwargs):
        query = session.query(cls)
        if kwargs:
            # Apply filters based on kwargs
            for key, value in kwargs.items():
                if hasattr(cls, key):
                    query = query.filter(getattr(cls, key) == value)
                else:
                    raise ValueError(f"Invalid filter key '{key}' for model {cls.__name__}")
        return query.all()

    @classmethod
    def get_by_id(cls, session, id):
        return session.query(cls).filter(cls.id == id).first()

    @classmethod
    def update(cls, session, id, **kwargs):
        instance = cls.get_by_id(session, id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
                session.commit()
                session.refresh(instance)
        return instance
    
    @classmethod
    def delete(cls, session, id):
        instance = cls.get_by_id(session, id)
        if instance:
            session.delete(instance)
            session.commit()
            return True
        return False



class Chat(Base, CRUDMixin):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[Optional[str]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    
    stages = relationship("Stage", back_populates="chat")


class Message(Base, CRUDMixin):
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    stage_id: Mapped[int] = mapped_column(Integer, ForeignKey('stages.id'), nullable=False)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))

    stage: Mapped["Stage"] = relationship("Stage", back_populates="messages")

    def get_role_display(self) -> str:
        """Get the display version of the role."""
        return MessageRole.get_display(self.role)


class Stage(Base, CRUDMixin):
    __tablename__ = "stages"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[Optional[str]] = mapped_column(nullable=True)
    status: Mapped[StageStatus] = mapped_column(nullable=False, default=StageStatus.NOT_STARTED)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    next_stage_id: Mapped[Optional[int]] = mapped_column(ForeignKey('stages.id'), nullable=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey('chats.id'), nullable=False)
    pipeline_endpoint: Mapped[Optional[str]] = mapped_column(nullable=True)

    chat: Mapped["Chat"] = relationship("Chat", back_populates="stages")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="stage")
    next_stage: Mapped[Optional["Stage"]] = relationship(
        "Stage", 
        remote_side=[id], 
        backref="previous_stage", 
        uselist=False
    )

    def get_status_display(self) -> str:
        """Get the display version of the status."""
        return StageStatus.get_display(self.status)

    @classmethod
    def create_default_stages(cls, session, chat_id) -> list["Stage"]:
        """Create the default stages for a chat and link them sequentially."""
        stages = [
            {
                "name": "Requirements Gathering", 
                "description": "Gathering requirements from the user",
                "pipeline_endpoint": "/api/pipeline/requirements-gatherer"
            },
            {
                "name": "Technical Design", 
                "description": "Designing the technical solution",
                "pipeline_endpoint": "/api/pipeline/tech-spec-generator"
            },
            {
                "name": "Implementation", 
                "description": "Implementing the solution",
                "pipeline_endpoint": "/api/pipeline/implementation"
            },
            {
                "name": "Code Review", 
                "description": "Create the MR and iterate based on feedback",
                "pipeline_endpoint": "/api/pipeline/code-review"
            },
        ]
        previous_stage = None
        new_stages = []
        for stage in stages:
            new_stage = cls.create(session, chat_id=chat_id, **stage)
            if previous_stage:
                previous_stage.next_stage_id = new_stage.id
                session.commit()
                session.refresh(previous_stage)
            new_stages.append(new_stage)
            previous_stage = new_stage

        return new_stages

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context manager for database sessions.
    
    Usage:
        with get_db() as session:
            # perform database operations
            session.query(Model).all()
            # commit is handled automatically
            # rollback on exception is handled automatically
            # session is closed automatically
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
