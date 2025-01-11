from code_assistant.embedding.models.models import EmbeddingModel
from code_assistant.logging.logger import get_logger
from code_assistant.storage.codebase import Class
from code_assistant.storage.stores import CodeStore

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
        code_store: CodeStore,
    ):
        """
        Generate embeddings for a codebase.
        """

        for file in code_store:
            for unit in file:
                try:
                    formatted_string = (
                        f"type: {unit.unit_type}, "
                        f"name: {unit.name}, "
                        f"filepath: {unit.file.filepath}, "
                        f"source_code: {unit.source_code}"
                    )
                    unit.embeddings[self.model.model_name] = (
                        self.model.generate_embedding(formatted_string)
                    )
                    code_store.save_unit(unit)

                    if isinstance(unit, Class):
                        for method in unit.methods:
                            formatted_string = (
                                f"type: {method.unit_type}, "
                                f"filepath: {method.class_ref.file.filepath}, "
                                f"class: {method.class_ref.name}, "
                                f"name: {method.name}, "
                                f"source_code: {method.source_code}"
                            )
                            method.embeddings[self.model.model_name] = (
                                self.model.generate_embedding(formatted_string)
                            )
                            code_store.save_unit(method)

                except Exception as e:
                    logger.error(f"Failed to embed unit {unit.name}: {e}")
