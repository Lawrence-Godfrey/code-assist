from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.storage.codebase import Method
from code_assistant.storage.stores.code import MongoDBCodeStore

logger = get_logger(__name__)


class CodeEmbedder:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize code embedder which generates embeddings for code units and queries.

        Args:
            embedding_model (EmbeddingModel): Embedding model to use for generating embeddings
        """
        self.model = embedding_model
        self.embedding_dimension = self.model.embedding_dimension

    def embed_code_units(
        self,
        code_store: MongoDBCodeStore,
    ) -> int:
        """
        Generate embeddings for a codebase.

        Args:
            code_store (MongoDBCodeStore): Store containing code units to embed

        Returns:
            Number of code units processed
        """

        processed_units = 0

        for unit in code_store:
            try:
                formatted_string = (
                    f"type: {unit.unit_type}, "
                    f"name: {unit.name}, "
                    f"filepath: {unit.filepath}, "
                    f"source_code: {unit.source_code}"
                )
                if isinstance(unit, Method):
                    formatted_string += f"\nclassname: {unit.classname}"

                unit.embeddings[self.model.model_name] = self.model.generate_embedding(
                    formatted_string
                )

                code_store.save_item(unit)

                processed_units += 1

            except Exception as e:
                logger.error(f"Failed to embed unit {unit.name}: {e}")

        return processed_units
