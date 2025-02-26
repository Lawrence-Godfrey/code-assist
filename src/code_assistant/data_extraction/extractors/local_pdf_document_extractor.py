"""
Extract documents from PDF files.

This module provides functionality to extract text content from PDF files,
convert it to Markdown format, and store it in MongoDB for use in the
document storage system.
"""

from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Union

from pypdf import PdfReader

from code_assistant.data_extraction.extractors.exceptions import SourceExistsError
from code_assistant.logging.logger import get_logger
from code_assistant.storage.document import PDFDocument
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class PDFDocumentExtractor:
    """Extract text content from PDF files."""

    def _extract_text_from_pdf(self, file_path: Path) -> tuple[str, int]:
        """
        Extract text content from a PDF file and convert to markdown.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (extracted text in Markdown format, page count)

        Raises:
            ValueError: If the file cannot be read or parsed
        """
        try:
            with open(file_path, "rb") as file:
                # Create PDF reader object
                reader = PdfReader(file)
                page_count = len(reader.pages)

                content = []

                # Extract text from each page
                for i, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        content.append(f"## Page {i}\n\n{text}\n")

                return "\n".join(content), page_count

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def _get_pdf_files(self, path: Union[str, Path]) -> Iterator[Path]:
        """
        Get all PDF files from a path.

        Args:
            path: Path to a PDF file or directory containing PDFs

        Returns:
            Iterator of Path objects for PDF files
        """
        path = Path(path)

        if path.is_file():
            if path.suffix.lower() == ".pdf":
                yield path
        else:
            for pdf_file in path.rglob("*.pdf"):
                yield pdf_file

    async def extract_documents(
        self,
        path: Union[str, Path],
        doc_store: MongoDBDocumentStore,
        collection_name: str,
        overwrite: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        """
        Extract text from PDF files and store as documents.

        Args:
            path: Path to PDF file or directory containing PDFs
            doc_store: Document store to save extracted documents
            collection_name: A name describing the collection to extract from
            overwrite: Whether to overwrite existing source content
            limit: Optional limit on number of files to process

        Raises:
            SourceExistsError: If source exists and overwrite is False
            ValueError: If path doesn't exist or files can't be accessed
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Check if source already exists in storage
        if doc_store.namespace_exists() and not overwrite:
            raise SourceExistsError(
                f"Source {collection_name} already exists. Use overwrite=True to replace."
            )

        if overwrite:
            logger.info(f"Overwriting source {collection_name}")
            doc_store.delete_namespace()

        try:
            files_processed = 0

            # Process each PDF file
            for pdf_path in self._get_pdf_files(path):
                if limit and files_processed >= limit:
                    break

                try:
                    # Extract text content
                    content, page_count = self._extract_text_from_pdf(pdf_path)

                    # Create document object
                    document = PDFDocument(
                        title=pdf_path.stem,
                        # Use filename without extension as title
                        content=content,
                        source_id=collection_name,
                        last_modified=datetime.fromtimestamp(pdf_path.stat().st_mtime),
                        file_path=str(pdf_path.absolute()),
                        page_count=page_count,
                        file_size=pdf_path.stat().st_size,
                    )

                    # Save to document store
                    doc_store.save_item(document)
                    files_processed += 1

                    logger.info(f"Processed PDF: {pdf_path.name}")

                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
                    continue

            logger.info(
                f"Processed {files_processed} PDF files in source {collection_name}"
            )

        except Exception as e:
            logger.error(f"Error during PDF extraction: {str(e)}")
            raise
