import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

from embedding.generate_embeddings import CodeEmbedder
from embedding.compare_embeddings import EmbeddingSimilaritySearch
from storage.code_store import CodebaseSnapshot
from evaluation.data_generators.prompt_code_pair_dataset import PromptCodePairDataset
from embedding.models.models import (
    EmbeddingModelFactory,
    EmbeddingModel,
    OpenAIEmbeddingModel,
)


@dataclass
class RetrievalMetrics:
    """Stores various retrieval evaluation metrics."""

    model_name: str
    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hits_at_k: Dict[int, float]
    mean_avg_precision: float
    mean_similarity_correct: float
    mean_similarity_incorrect: float

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary format."""
        return {
            "model_name": self.model_name,
            "mrr": self.mrr,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hits_at_k": self.hits_at_k,
            "mean_average_precision": self.mean_avg_precision,
            "mean_similarity_correct": self.mean_similarity_correct,
            "mean_similarity_incorrect": self.mean_similarity_incorrect,
        }


class CodeRetrievalEvaluator:
    """Evaluates code retrieval performance using various metrics."""

    def __init__(
        self,
        embedder: CodeEmbedder,
        test_dataset: PromptCodePairDataset,
        similarity_engine: EmbeddingSimilaritySearch,
        k_values: List[int] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            embedder: Initialized CodeEmbedder instance
            test_dataset: Dataset of prompt-code pairs for testing
            similarity_engine: Initialized EmbeddingSimilaritySearch instance
            k_values: List of K values for computing metrics (default: [1, 3, 5, 10])
        """
        self.embedder = embedder
        self.test_dataset = test_dataset
        self.searcher = similarity_engine
        self.k_values = k_values or [1, 3, 5, 10]
        self.max_k = max(self.k_values)

    def _compute_retrieval_position(
        self, retrieved_ids: List[str], correct_id: str
    ) -> Optional[int]:
        """Find position of correct code unit in retrieved results (1-based)."""
        try:
            return retrieved_ids.index(correct_id) + 1
        except ValueError:
            return None

    def _compute_mrr(self, ranks: List[Optional[int]]) -> float:
        """Compute Mean Reciprocal Rank."""
        reciprocal_ranks = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
        return float(np.mean(reciprocal_ranks))

    def _compute_precision_at_k(
        self, retrieved_ids: List[str], correct_id: str, k: int
    ) -> float:
        """Compute Precision@K."""
        retrieved_at_k = retrieved_ids[:k]
        return 1.0 if correct_id in retrieved_at_k else 0.0

    def _compute_average_precision(
        self, retrieved_ids: List[str], correct_id: str
    ) -> float:
        """Compute Average Precision for a single query."""
        if correct_id not in retrieved_ids:
            return 0.0

        precisions = []
        for k in range(1, len(retrieved_ids) + 1):
            if retrieved_ids[k - 1] == correct_id:
                precision_at_k = self._compute_precision_at_k(
                    retrieved_ids, correct_id, k
                )
                precisions.append(precision_at_k)

        return np.mean(precisions) if precisions else 0.0

    def evaluate(self) -> RetrievalMetrics:
        """
        Evaluate retrieval performance on test dataset.

        Returns:
            RetrievalMetrics containing various evaluation metrics
        """
        ranks = []
        precisions_at_k = {k: [] for k in self.k_values}
        hits_at_k = {k: [] for k in self.k_values}
        avg_precisions = []

        similarities_correct = []
        similarities_incorrect = []

        # Evaluate each prompt-code pair
        for pair in tqdm(self.test_dataset, desc="Evaluating retrieval"):
            # Generate prompt embedding
            prompt_embedding = self.embedder.model.generate_embedding(pair.prompt)

            # Get top K results
            results = self.searcher.find_similar(prompt_embedding, self.max_k)
            retrieved_ids = [r.code_unit.id for r in results]

            # Record similarity scores
            for result in results:
                if result.code_unit.id == pair.code_unit.id:
                    similarities_correct.append(result.similarity_score)
                else:
                    similarities_incorrect.append(result.similarity_score)

            # Compute rank
            rank = self._compute_retrieval_position(retrieved_ids, pair.code_unit.id)
            ranks.append(rank)

            # Compute precision@k and hits@k for each k
            for k in self.k_values:
                precision = self._compute_precision_at_k(
                    retrieved_ids, pair.code_unit.id, k
                )
                precisions_at_k[k].append(precision)
                hits_at_k[k].append(1.0 if precision > 0 else 0.0)

            # Compute average precision
            avg_precision = self._compute_average_precision(
                retrieved_ids, pair.code_unit.id
            )
            avg_precisions.append(avg_precision)

        # Aggregate metrics
        metrics = RetrievalMetrics(
            model_name=self.embedder.model.model_name,
            mrr=self._compute_mrr(ranks),
            precision_at_k={
                k: float(np.mean(precisions))
                for k, precisions in precisions_at_k.items()
            },
            recall_at_k={
                k: float(np.mean(precisions))
                for k, precisions in precisions_at_k.items()
            },
            hits_at_k={k: float(np.mean(hits)) for k, hits in hits_at_k.items()},
            mean_avg_precision=float(np.mean(avg_precisions)),
            mean_similarity_correct=float(np.mean(similarities_correct)),
            mean_similarity_incorrect=float(np.mean(similarities_incorrect)),
        )

        return metrics


class MultiModelCodeRetrievalEvaluator:
    """Evaluates code retrieval performance across multiple embedding models."""

    def __init__(
        self,
        test_dataset: PromptCodePairDataset,
        codebase: CodebaseSnapshot,
        models: List[EmbeddingModel],
        k_values: List[int] = None,
    ):
        """
        Initialize the multi-model evaluator.

        Args:
            test_dataset: Dataset of prompt-code pairs for testing
            codebase: Full codebase with embedded code units
            models: List of embedding models to evaluate
            k_values: List of K values for computing metrics
        """
        self.test_dataset = test_dataset
        self.codebase = codebase
        self.k_values = k_values
        self.models = models

    def evaluate_all_models(self) -> List[RetrievalMetrics]:
        """
        Evaluate retrieval performance using all available embedding models.

        Returns:
            List of RetrievalMetrics, one for each model
        """
        metrics_list = []

        for model in self.models:
            try:

                print(f"\nEvaluating model: {model.model_name}")

                # Initialize embedder for current model
                embedder = CodeEmbedder(embedding_model=model)

                similarity_engine = EmbeddingSimilaritySearch(
                    codebase=self.codebase, embedding_model=model
                )

                # Create and run evaluator for current model
                evaluator = CodeRetrievalEvaluator(
                    embedder=embedder,
                    test_dataset=self.test_dataset,
                    similarity_engine=similarity_engine,
                    k_values=self.k_values,
                )

                metrics = evaluator.evaluate()
                metrics_list.append(metrics)

            except Exception as e:
                print(f"Error evaluating {model.model_name}: {str(e)}")
                continue

        return metrics_list

    def print_comparison(self, metrics_list: List[RetrievalMetrics]) -> None:
        """Print a comparison of metrics across all models."""
        print("\nModel Comparison Results:")
        print("=" * 80)

        # Print MRR comparison
        print("\nMean Reciprocal Rank:")
        for metrics in metrics_list:
            print(f"{metrics.model_name}: {metrics.mrr:.3f}")

        # Print Precision@K comparison
        print("\nPrecision@K:")
        k_values = list(metrics_list[0].precision_at_k.keys())
        for k in k_values:
            print(f"\nP@{k}:")
            for metrics in metrics_list:
                print(f"{metrics.model_name}: {metrics.precision_at_k[k]:.3f}")

        # Print Mean Average Precision comparison
        print("\nMean Average Precision:")
        for metrics in metrics_list:
            print(f"{metrics.model_name}: {metrics.mean_avg_precision:.3f}")

        # Print similarity score analysis
        print("\nSimilarity Score Analysis:")
        for metrics in metrics_list:
            print(f"\n{metrics.model_name}:")
            print(f"  Correct matches: {metrics.mean_similarity_correct:.3f}")
            print(f"  Incorrect matches: {metrics.mean_similarity_incorrect:.3f}")
