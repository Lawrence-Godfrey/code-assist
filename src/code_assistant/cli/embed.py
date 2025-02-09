import os
from enum import Enum
from typing import Optional, Union

from code_assistant.embedding.code_embedder import CodeEmbedder
from code_assistant.embedding.document_embedder import DocumentEmbedder
from code_assistant.logging.logger import get_logger
from code_assistant.models.factory import ModelFactory
from code_assistant.storage.stores.code import MongoDBCodeStore
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class StoreType(Enum):
    """Type of store to embed."""

    CODE = "code"
    DOCUMENT = "document"


class EmbedCommands:
    """Commands for generating and comparing embeddings."""

    def generate(
        self,
        namespace: str,
        store_type: str = "code",
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = ModelFactory.get_default_embedding_model(),
    ) -> None:
        """
        Generate embeddings for code units.

        Args:
            namespace: Name of the namespace (codebase or space_key)
            store_type: Type of store to embed ('code' or 'document')
            database_url: MongoDB connection URL
            model_name: Name of the embedding model to use
        """
        try:
            store_type = StoreType(store_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid store type: {store_type}. Must be 'code' or 'document'"
            )

        database_url = os.getenv("MONGODB_URL") or database_url

        self._process_embeddings(
            namespace=namespace,
            database_url=database_url,
            model_name=model_name,
            store_type=store_type,
        )

    def _process_embeddings(
        self,
        namespace: str,
        database_url: str,
        model_name: str,
        store_type: StoreType,
    ) -> None:
        """
        Generate embeddings for code units from a codebase.

        Args:
            namespace: Name of the namespace (codebase or space_key).
            database_url: URL for MongoDB database
            model_name: Name of the model to use for embeddings
            store_type: Type of store to process (code or document)
        """

        store = self._setup_store(namespace, database_url, store_type)
        embedding_model = ModelFactory.create(model_name)

        # Generate embeddings
        logger.info("Generating embeddings...")

        if store_type == StoreType.CODE:
            embedder = CodeEmbedder(embedding_model=embedding_model)
            items_processed = embedder.embed_code_units(store)
        else:
            embedder = DocumentEmbedder(embedding_model=embedding_model)
            items_processed = embedder.embed_documents(store)

        store.refresh_vector_indexes()

        # Print statistics
        logger.info(f"Total items processed: {items_processed}")
        logger.info(f"Embedding dimension: {embedding_model.embedding_dimension}")

    def _setup_store(
        self,
        namespace: str,
        database_url: str,
        store_type: StoreType,
    ) -> Union[MongoDBCodeStore, MongoDBDocumentStore]:
        """Set up the appropriate store based on type."""
        logger.info(f"Loading items from {database_url}...")

        if store_type == StoreType.CODE:
            store = MongoDBCodeStore(codebase=namespace, connection_string=database_url)
        else:
            store = MongoDBDocumentStore(
                space_key=namespace, connection_string=database_url
            )

        if not store.namespace_exists():
            raise ValueError(f"Namespace {namespace} does not exist.")

        return store
