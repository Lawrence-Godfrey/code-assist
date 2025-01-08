from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, List, Optional
import os

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from code_assistant.storage.code_store import (
    CodeUnit,
    CodeEmbedding,
    CodebaseSnapshot,
    File,
)
from code_assistant.logging.logger import get_logger

logger = get_logger(__name__)


class DatabaseConfig:
    """Configuration for database connections."""

    def __init__(
        self,
        connection_string: str = None,
        database_name: str = "code_assistant",
    ):
        """
        Initialize database configuration.

        Args:
            connection_string: Database connection string. If None, uses DATABASE_URI env var
            database_name: Name of the database to use
        """
        self.connection_string = connection_string or os.getenv("DATABASE_URI")
        self.database_name = database_name


class CodebaseStorage(ABC):
    """Abstract interface for codebase storage operations."""

    @abstractmethod
    def save_code_unit(self, code_unit: CodeUnit) -> str:
        """Save a code unit to storage and return its ID."""
        pass

    @abstractmethod
    def get_code_unit(self, unit_id: str) -> Optional[CodeUnit]:
        """Retrieve a code unit by ID."""
        pass

    @abstractmethod
    def get_code_units_by_type(self, unit_type: str) -> List[CodeUnit]:
        """Retrieve all code units of a specific type."""
        pass

    @abstractmethod
    def update_code_unit(self, unit_id: str, code_unit: CodeUnit) -> bool:
        """Update an existing code unit."""
        pass

    @abstractmethod
    def delete_code_unit(self, unit_id: str) -> bool:
        """Delete a code unit by ID."""
        pass

    @abstractmethod
    def save_codebase(self, codebase: CodebaseSnapshot) -> None:
        """Save entire codebase snapshot."""
        pass

    @abstractmethod
    def load_codebase(self) -> CodebaseSnapshot:
        """Load entire codebase snapshot."""
        pass


class VectorStorage(ABC):
    """Abstract interface for vector embedding storage and search."""

    @abstractmethod
    def save_embedding(self, unit_id: str, embedding: CodeEmbedding) -> None:
        """Save a vector embedding for a code unit."""
        pass

    @abstractmethod
    def get_embedding(self, unit_id: str, model_name: str) -> Optional[CodeEmbedding]:
        """Retrieve embedding for a code unit by model name."""
        pass

    @abstractmethod
    def find_similar(
        self,
        query_embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Find similar code units using vector similarity search."""
        pass

    @abstractmethod
    def delete_embedding(self, unit_id: str, model_name: str) -> bool:
        """Delete embedding for a code unit."""
        pass


class MongoCodebaseStorage(CodebaseStorage):
    """MongoDB implementation of codebase storage."""

    def __init__(self, config: DatabaseConfig):
        """Initialize MongoDB connection for codebase storage."""
        self.client = MongoClient(config.connection_string)
        self.db: Database = self.client[config.database_name]
        self.code_units: Collection = self.db.code_units

        # Create indexes
        self.code_units.create_index("id", unique=True)
        self.code_units.create_index("unit_type")

    def _serialize_code_unit(self, code_unit: CodeUnit) -> dict:
        """Convert code unit to MongoDB document."""
        doc = asdict(code_unit)
        # Remove embeddings as they'll be stored separately
        doc.pop("embeddings", None)
        return doc

    def _deserialize_code_unit(self, doc: dict) -> CodeUnit:
        """Convert MongoDB document back to code unit."""
        unit_type = doc.pop("unit_type")
        # Import dynamically to avoid circular imports
        from code_assistant.storage.code_store import Class, Method, Function, File

        type_mapping = {
            "class": Class,
            "method": Method,
            "function": Function,
            "file": File,
        }

        cls = type_mapping.get(unit_type)
        if not cls:
            raise ValueError(f"Unknown unit type: {unit_type}")

        return cls(**doc)

    def save_code_unit(self, code_unit: CodeUnit) -> str:
        """Save a code unit to MongoDB."""
        doc = self._serialize_code_unit(code_unit)
        result = self.code_units.update_one(
            {"id": code_unit.id}, {"$set": doc}, upsert=True
        )
        return str(code_unit.id)

    def get_code_unit(self, unit_id: str) -> Optional[CodeUnit]:
        """Retrieve a code unit from MongoDB."""
        doc = self.code_units.find_one({"id": unit_id})
        if doc:
            return self._deserialize_code_unit(doc)
        return None

    def get_code_units_by_type(self, unit_type: str) -> List[CodeUnit]:
        """Retrieve code units of a specific type from MongoDB."""
        docs = self.code_units.find({"unit_type": unit_type})
        return [self._deserialize_code_unit(doc) for doc in docs]

    def update_code_unit(self, unit_id: str, code_unit: CodeUnit) -> bool:
        """Update a code unit in MongoDB."""
        doc = self._serialize_code_unit(code_unit)
        result = self.code_units.update_one({"id": unit_id}, {"$set": doc})
        return result.modified_count > 0

    def delete_code_unit(self, unit_id: str) -> bool:
        """Delete a code unit from MongoDB."""
        result = self.code_units.delete_one({"id": unit_id})
        return result.deleted_count > 0

    def save_codebase(self, codebase: CodebaseSnapshot) -> None:
        """Save entire codebase to MongoDB."""
        for unit in codebase.iter_flat():
            self.save_code_unit(unit)

    def load_codebase(self) -> CodebaseSnapshot:
        """Load entire codebase from MongoDB."""
        codebase = CodebaseSnapshot()
        docs = self.code_units.find({})

        # First pass: Create all code units
        units_by_id = {}
        for doc in docs:
            unit = self._deserialize_code_unit(doc)
            units_by_id[unit.id] = unit

        # Second pass: Establish relationships
        for unit_id, unit in units_by_id.items():
            if hasattr(unit, "file_id") and unit.file_id:
                unit.file = units_by_id.get(unit.file_id)
            if hasattr(unit, "class_id") and unit.class_id:
                unit.class_ref = units_by_id.get(unit.class_id)

        # Add to codebase
        for unit in units_by_id.values():
            if isinstance(unit, File):
                codebase.add_file(unit)

        return codebase


class MongoVectorStorage(VectorStorage):
    """MongoDB implementation of vector storage with vector search capabilities."""

    def __init__(self, config: DatabaseConfig):
        """Initialize MongoDB connection for vector storage."""
        self.client = MongoClient(config.connection_string)
        self.db: Database = self.client[config.database_name]
        self.embeddings: Collection = self.db.embeddings

        # Create indexes
        self.embeddings.create_index([("unit_id", 1), ("model_name", 1)], unique=True)
        # Create vector index for similarity search
        self.embeddings.create_index(
            [("vector", "vector"), ("model_name", 1)],
            {
                "vectorSize": 1536,  # Default size for OpenAI embeddings
                "vectorSearchOptions": {"similarity": "cosine"},
            },
        )

    def save_embedding(self, unit_id: str, embedding: CodeEmbedding) -> None:
        """Save vector embedding to MongoDB."""
        doc = {
            "unit_id": unit_id,
            "model_name": embedding.model_name,
            "vector": embedding.vector.tolist(),
        }
        self.embeddings.update_one(
            {"unit_id": unit_id, "model_name": embedding.model_name},
            {"$set": doc},
            upsert=True,
        )

    def get_embedding(self, unit_id: str, model_name: str) -> Optional[CodeEmbedding]:
        """Retrieve embedding from MongoDB."""
        doc = self.embeddings.find_one({"unit_id": unit_id, "model_name": model_name})
        if doc:
            return CodeEmbedding(vector=doc["vector"], model_name=doc["model_name"])
        return None

    def find_similar(
        self,
        query_embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Find similar vectors using MongoDB's vector search."""
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding.vector.tolist(),
                    "path": "vector",
                    "numCandidates": top_k
                    * 10,  # Search among more candidates for better results
                    "limit": top_k,
                    "index": "vector_index",
                }
            }
        ]

        if threshold:
            pipeline.append({"$match": {"score": {"$gte": threshold}}})

        results = list(self.embeddings.aggregate(pipeline))
        return [
            {
                "unit_id": r["unit_id"],
                "score": r["score"],
                "model_name": r["model_name"],
            }
            for r in results
        ]

    def delete_embedding(self, unit_id: str, model_name: str) -> bool:
        """Delete embedding from MongoDB."""
        result = self.embeddings.delete_one(
            {"unit_id": unit_id, "model_name": model_name}
        )
        return result.deleted_count > 0


class StorageFactory:
    """Factory for creating storage instances based on database URLs."""

    # Map of URL schemes to storage implementations
    STORAGE_TYPES = {
        "mongodb": (MongoCodebaseStorage, MongoVectorStorage),
        "mongodb+srv": (MongoCodebaseStorage, MongoVectorStorage),  # MongoDB Atlas
        "postgresql": None,  # Future implementation
        "mysql": None,  # Future implementation
        "redis": None,  # Future implementation
        "pinecone": None,  # Future implementation for vector storage
        "milvus": None,  # Future implementation for vector storage
        "weaviate": None,  # Future implementation for vector storage
    }

    @classmethod
    def get_storage_type_from_url(cls, url: str) -> str:
        """
        Extract the database type from a connection URL.

        Args:
            url: Database connection URL
                Examples:
                - mongodb://localhost:27017
                - mongodb+srv://user:pass@cluster.mongodb.net/
                - postgresql://user:pass@localhost/dbname
                - redis://localhost:6379
                - pinecone://api-key@index-name

        Returns:
            Database type as string

        Raises:
            ValueError: If URL scheme is not supported or URL is invalid
        """
        try:
            scheme = url.split("://")[0].lower()
            if scheme not in cls.STORAGE_TYPES:
                supported = ", ".join(cls.STORAGE_TYPES.keys())
                raise ValueError(
                    f"Unsupported database type: {scheme}. "
                    f"Supported types are: {supported}"
                )
            return scheme
        except IndexError:
            raise ValueError(f"Invalid database URL format: {url}")

    @classmethod
    def create_codebase_storage(cls, config: DatabaseConfig) -> CodebaseStorage:
        """
        Create a codebase storage instance based on the connection URL.

        Args:
            config: Database configuration containing connection URL

        Returns:
            Appropriate CodebaseStorage implementation

        Raises:
            ValueError: If database type is not supported
        """
        storage_type = cls.get_storage_type_from_url(config.connection_string)
        implementation = cls.STORAGE_TYPES.get(storage_type)

        if not implementation:
            raise ValueError(f"No implementation available for {storage_type}")

        codebase_storage_class = implementation[0]
        return codebase_storage_class(config)

    @classmethod
    def create_vector_storage(cls, config: DatabaseConfig) -> VectorStorage:
        """
        Create a vector storage instance based on the connection URL.

        Args:
            config: Database configuration containing connection URL

        Returns:
            Appropriate VectorStorage implementation

        Raises:
            ValueError: If database type is not supported
        """
        storage_type = cls.get_storage_type_from_url(config.connection_string)
        implementation = cls.STORAGE_TYPES.get(storage_type)

        if not implementation:
            raise ValueError(f"No implementation available for {storage_type}")

        vector_storage_class = implementation[1]
        return vector_storage_class(config)
