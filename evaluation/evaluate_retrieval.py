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


@dataclass
class RetrievalMetrics:
    """Stores various retrieval evaluation metrics."""

    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    hits_at_k: Dict[int, float]
    mean_avg_precision: float
    mean_similarity_correct: float
    mean_similarity_incorrect: float


class CodeRetrievalEvaluator:
    """Evaluates code retrieval performance using various metrics."""

    def __init__(
        self,
        embedder: CodeEmbedder,
        test_dataset: PromptCodePairDataset,
        codebase: CodebaseSnapshot,
        k_values: List[int] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            embedder: Initialized CodeEmbedder instance
            test_dataset: Dataset of prompt-code pairs for testing
            codebase: Full codebase with embedded code units
            k_values: List of K values for computing metrics (default: [1, 3, 5, 10])
        """
        self.embedder = embedder
        self.test_dataset = test_dataset
        self.searcher = EmbeddingSimilaritySearch(codebase)
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


def main(
    test_data_path: str,
    codebase_path: str,
    embedding_model: str = "jinaai/jina-embeddings-v3",
    output_path: Optional[str] = None,
) -> None:
    """
    Run evaluation and save results.

    Args:
        test_data_path: Path to test dataset JSON
        codebase_path: Path to embedded codebase JSON
        embedding_model: Name of embedding model to use
        output_path: Optional path to save evaluation results
    """
    # Load test dataset and codebase
    test_dataset = PromptCodePairDataset.from_json(
        Path(test_data_path), CodebaseSnapshot.from_json(Path(codebase_path))
    )

    codebase = CodebaseSnapshot.from_json(Path(codebase_path))

    # Initialize embedder and evaluator
    embedder = CodeEmbedder(embedding_model=embedding_model)
    evaluator = CodeRetrievalEvaluator(embedder, test_dataset, codebase)

    # Run evaluation
    metrics = evaluator.evaluate()

    # Print results
    print("\nEvaluation Results:")
    print(f"Mean Reciprocal Rank: {metrics.mrr:.3f}")
    print("\nPrecision@K:")
    for k, value in metrics.precision_at_k.items():
        print(f"  P@{k}: {value:.3f}")
    print("\nHits@K:")
    for k, value in metrics.hits_at_k.items():
        print(f"  H@{k}: {value:.3f}")
    print(f"\nMean Average Precision: {metrics.mean_avg_precision:.3f}")
    print("\nSimilarity Analysis:")
    print(f"  Mean Similarity (Correct): {metrics.mean_similarity_correct:.3f}")
    print(f"  Mean Similarity (Incorrect): {metrics.mean_similarity_incorrect:.3f}")

    # Save results if output path provided
    if output_path:
        results = {
            "mrr": metrics.mrr,
            "precision_at_k": metrics.precision_at_k,
            "recall_at_k": metrics.recall_at_k,
            "hits_at_k": metrics.hits_at_k,
            "mean_average_precision": metrics.mean_avg_precision,
            "mean_similarity_correct": metrics.mean_similarity_correct,
            "mean_similarity_incorrect": metrics.mean_similarity_incorrect,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
