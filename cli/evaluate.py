from pathlib import Path
from typing import Optional

from evaluation.evaluate_retrieval import MultiModelCodeRetrievalEvaluator
from evaluation.data_generators.prompt_code_pair_dataset import PromptCodePairDataset
from storage.code_store import CodebaseSnapshot
from embedding.models.models import EmbeddingModelFactory, OpenAIEmbeddingModel


class EvaluateCommands:
    """Commands for evaluating retrieval performance."""

    def retrieval(
        self,
        test_data_path: str,
        codebase_path: str,
        output_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Evaluate retrieval performance across models."""
        test_dataset = PromptCodePairDataset.from_json(
            Path(test_data_path), CodebaseSnapshot.from_json(Path(codebase_path))
        )
        codebase = CodebaseSnapshot.from_json(Path(codebase_path))

        models = []
        model_classes = EmbeddingModelFactory.models()
        for model_name, cls in model_classes.items():
            if issubclass(cls, OpenAIEmbeddingModel):
                if openai_api_key:
                    model = cls(model_name=model_name, api_key=openai_api_key)
                    models.append(model)
            else:
                model = cls(model_name=model_name)
                models.append(model)

        evaluator = MultiModelCodeRetrievalEvaluator(
            test_dataset=test_dataset, codebase=codebase, models=models
        )

        metrics_list = evaluator.evaluate_all_models()
        evaluator.print_comparison(metrics_list)
