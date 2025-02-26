from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class DocumentEmbedder:
    """Class for generating embeddings for documents."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize document embedder which generates embeddings for documents.

        Args:
            embedding_model: Model to use for generating embeddings
        """
        self.model = embedding_model
        self.embedding_dimension = self.model.embedding_dimension

    def embed_documents(
        self,
        document_store: MongoDBDocumentStore,
    ) -> int:
        """
        Generate embeddings for all documents in a store.

        Args:
            document_store: Store containing documents to embed

        Returns:
            Number of documents processed
        """
        processed_documents = 0

        for document in document_store:
            try:
                # Get text representation of document for embedding
                text_for_embedding = document.get_full_text()

                # Generate and store embedding
                document.embeddings[self.model.model_name] = (
                    self.model.generate_embedding(text_for_embedding)
                )

                document_store.save_item(document)
                processed_documents += 1

            except Exception as e:
                logger.error(f"Failed to embed document {document.title}: {e}")

        return processed_documents
