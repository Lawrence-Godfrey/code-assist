import dotenv
import fire

from code_assistant.cli import (
    context,
    embed,
    evaluate,
    extract,
    generate,
    pipeline,
    rag,
)

dotenv.load_dotenv()


class CLI:
    """Main CLI interface for the code assistant package."""

    def __init__(self):
        self.context = context.ContextCommands()
        self.extract = extract.ExtractCommands()
        self.embed = embed.EmbedCommands()
        self.evaluate = evaluate.EvaluateCommands()
        self.generate = generate.GenerateCommands()
        self.rag = rag.RagCommands()
        self.pipeline = pipeline.PipelineCommands()


def main():
    """Entry point for the CLI."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
