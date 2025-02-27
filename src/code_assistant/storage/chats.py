import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from datetime import datetime
from contextlib import contextmanager
from enum import Enum as PyEnum

from code_assistant.interfaces.api.models import MessageRole
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



class Chat(Base, CRUDMixin):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True, unique=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    stages = relationship("Stage", back_populates="chat")


class Message(Base, CRUDMixin):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    stage_id = Column(Integer, ForeignKey('stages.id'), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    stage = relationship("Stage", back_populates="messages")

    def get_role_display(self):
        """Get the display version of the role."""
        return MessageRole.get_display(self.role)


class Stage(Base, CRUDMixin):
    __tablename__ = "stages"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    status: Mapped[StageStatus] = mapped_column(nullable=False, default=StageStatus.NOT_STARTED)
    created_at = Column(DateTime, default=datetime.utcnow)
    next_stage_id = Column(Integer, ForeignKey('stages.id'), nullable=True)
    chat_id = Column(Integer, ForeignKey('chats.id'), nullable=False)

    chat = relationship("Chat", back_populates="stages")
    messages = relationship("Message", back_populates="stage")
    next_stage = relationship("Stage", remote_side=[id], backref="previous_stage", uselist=False)

    def get_status_display(self) -> str:
        """Get the display version of the status."""
        return StageStatus.get_display(self.status)

    @classmethod
    def create_default_stages(cls, session, chat_id) -> list["Stage"]:
        """Create the default stages for a chat."""
        stages = [
            {"name": "Requirements Gathering", "description": "Gathering requirements from the user"},
            {"name": "Technical Design", "description": "Designing the technical solution"},
            {"name": "Implementation", "description": "Implementing the solution"},
            {"name": "Code Review", "description": "Create the MR and iterate based on feedback"},
        ]
        new_stages = []
        for stage in stages:
            new_stage = cls.create(session, chat_id=chat_id, **stage)
            new_stages.append(new_stage)

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
