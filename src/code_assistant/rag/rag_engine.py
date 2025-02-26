"""
This module implements a RAG (Retrieval Augmented Generation) system for code
assistance on a specific codebase. It combines embedding-based similarity search
with large language models to provide contextually relevant answers to queries.

The system works in three stages:
1. Retrieval: Finds relevant code units by computing similarity between the
   query and code embeddings
2. Augmentation: Enhances the original query with retrieved code context
3. Generation: Uses a large language model to generate detailed responses based
   on the augmented prompt

Usage:
    python rag_engine.py --query "How do I implement X?" \
                        --input_path "path/to/code_units.json" \
                        --model "gpt-4" \
                        --top_k 5 \
                        --threshold 0.5 \
                        --logging_enabled True

Requirements:
    - OpenAI API key in .env file
    - Pre-generated code embeddings in JSON format
"""

from dataclasses import dataclass
from typing import List, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.models.prompt import PromptModel
from code_assistant.storage.codebase import CodeUnit
from code_assistant.storage.document import Document
from code_assistant.storage.stores.code import MongoDBCodeStore
from code_assistant.storage.stores.document import MongoDBDocumentStore
from code_assistant.storage.types import SearchResult

logger = get_logger(__name__)


@dataclass
class RetrievedContext:
    """Container for retrieved code and document results."""

    code_units: List[SearchResult[CodeUnit]]
    documents: List[SearchResult[Document]]


class RAGEngine:
    def __init__(
        self,
        code_store: MongoDBCodeStore,
        doc_store: MongoDBDocumentStore,
        embedding_model: EmbeddingModel,
        prompt_model: PromptModel,
        top_k_code: Optional[int] = 5,
        top_k_docs: Optional[int] = 5,
        threshold: Optional[float] = None,
    ):
        """
        Initialises the RAG engine.

        Args:
            code_store: Store for accessing code units.
            doc_store: Store for accessing documentation.
            embedding_model: EmbeddingModel object used to generate the embeddings.
            prompt_model: PromptModel object used for generating prompts.
            top_k_code: Maximum number of code units to retrieve
            top_k_docs: Maximum number of documents to retrieve
            threshold: Minimum similarity score (0-1) required for retrieved items

        Raises:
           FileNotFoundError: If the code_units_path file cannot be found.
           ValueError: If an unsupported prompt_model is specified.
        """
        self.code_store = code_store
        self.doc_store = doc_store
        self._top_k_code = top_k_code
        self._top_k_docs = top_k_docs
        self._threshold = threshold
        self._embedding_model = embedding_model
        self._prompt_model = prompt_model

    def process(self, query: str) -> str:
        """
        Processes a query through the complete RAG pipeline.

        This method executes the full retrieval-augmentation-generation sequence:
        1. Retrieves similar code units and documentation based on the query
        2. Augments the query with the retrieved context
        3. Generates a response using the specified LLM

        Args:
            query: The user's prompt to which RAG will be applied

        Returns:
            The generated response that answers the query using relevant context.
        """
        context = self._retrieve(query)
        prompt = self._augment(query, context)
        response = self._generate(prompt)

        return response

    def _retrieve(self, query: str) -> RetrievedContext:
        """
        Retrieves relevant code units and documentation by computing similarity
        with the query.

        Args:
            query: The user's prompt which will be used to find similar items.

        Returns:
            RetrievedContext containing similar code units and documents
        """
        query_embedding = self._embedding_model.generate_embedding(query)

        # Find similar code units
        code_units = self.code_store.vector_search(
            query_embedding,
            self._embedding_model,
            top_k=self._top_k_code,
            threshold=self._threshold,
        )

        # Find similar documents
        documents = self.doc_store.vector_search(
            query_embedding,
            self._embedding_model,
            top_k=self._top_k_docs,
            threshold=self._threshold,
        )

        return RetrievedContext(code_units=code_units, documents=documents)

    def _augment(self, query: str, context: RetrievedContext) -> str:
        """
        Augments the original query with relevant code and documentation context.

        Format of the constructed prompt:
           Question: <original query>

           Here is the relevant code context from the codebase:

           [Code Unit 1] Type: <type> | File: <filepath>
           Class: <class_name>  # if applicable
           ```python
           <source_code>
           ```
           Documentation: <docstring>  # if available
           Similarity Score: <score>

           [Additional code units...]

           Here is the relevant documentation:

           [Document 1] Title: <title>
           Content:
           <document_content>
           Similarity Score: <score>

           [Additional documents...]

           Based on the code and documentation context above, please provide a
           detailed answer...

        Args:
            query: The user's prompt to augment
            context: Retrieved code units and documents

        Returns:
            The augmented prompt
        """
        prompt = f"Question: {query}\n\n"

        # Add code context if any was found
        if context.code_units:
            prompt += "Here is the relevant code context:\n\n"

            for i, result in enumerate(context.code_units, 1):
                unit = result.item
                prompt += f"[Code Unit {i}] "
                prompt += f"Type: {unit.unit_type}  |  "
                prompt += f"Full Name: {unit.fully_qualified_name()}\n"

                # Add source code
                prompt += "```python\n"
                prompt += f"{unit.source_code}\n"
                prompt += "```\n"

                # Add docstring if it exists and isn't empty
                if unit.docstring and unit.docstring.strip():
                    prompt += f"Documentation: {unit.docstring}\n"

                prompt += f"Similarity Score: {result.similarity_score:.2f}\n\n"

        # Add documentation context if any was found
        if context.documents:
            prompt += "Here is the relevant documentation:\n\n"

            for i, result in enumerate(context.documents, 1):
                doc = result.item
                prompt += f"[Document {i}] Title: {doc.title}\n"
                prompt += "Content:\n"
                # Truncate very long documents to maintain reasonable prompt length
                content = (
                    doc.content[:1000] + "..."
                    if len(doc.content) > 1000
                    else doc.content
                )
                prompt += f"{content}\n"
                prompt += f"Similarity Score: {result.similarity_score:.2f}\n\n"

        prompt += "Based on the code and documentation context above, please "
        prompt += "provide a detailed answer to the question. Reference both "
        prompt += "code examples and documentation where relevant.\n"

        logger.info("\nAugmented Prompt:\n")
        logger.info(prompt)

        return prompt

    def _generate(self, prompt) -> str:
        """
        Generates a response using the augmented prompt via the specified prompt
        model.

        Args:
            prompt (str): The augmented prompt.

        Returns:
            response_message (str): The generated response from the large LLM.
        """
        system_prompt = (
            "You are a helpful assistant with expertise in Python programming. "
            "When providing explanations, reference both code examples and "
            "documentation to give comprehensive answers."
        )

        response = self._prompt_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

        logger.info("\nGenerated Response:\n")
        logger.info(response)

        return response
