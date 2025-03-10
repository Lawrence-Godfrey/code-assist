import json
import os
from pathlib import Path
from typing import Optional

from code_assistant.evaluation.data_generators.prompt_code_pair_dataset import (
    PromptCodePairDataset,
)
from code_assistant.evaluation.evaluate_retrieval import (
    MultiModelCodeRetrievalEvaluator,
)
from code_assistant.models.factory import ModelFactory
from code_assistant.storage.stores.code import MongoDBCodeStore


class EvaluateCommands:
    """Commands for evaluating retrieval performance."""

    def retrieval(
        self,
        test_data_path: str,
        codebase: str,
        database_url: str = "mongodb://localhost:27017/",
        output_path: Optional[str] = None,
    ) -> None:
        """Evaluate retrieval performance across models."""

        database_url = os.getenv("MONGODB_URL") or database_url

        code_store = MongoDBCodeStore(codebase=codebase, connection_string=database_url)

        test_dataset = PromptCodePairDataset.from_json(Path(test_data_path), code_store)

        models = []
        model_classes = ModelFactory.models()
        for model_name, cls in model_classes.items():
            model = cls(model_name=model_name)
            models.append(model)

        evaluator = MultiModelCodeRetrievalEvaluator(
            test_dataset=test_dataset, code_store=code_store, models=models
        )

        metrics_list = evaluator.evaluate_all_models()
        evaluator.print_comparison(metrics_list)

        # Save results if output path provided
        if output_path:
            results = {
                "metrics": [metrics.to_dict() for metrics in metrics_list],
                "evaluated_at": str(Path(output_path).resolve()),
            }

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
