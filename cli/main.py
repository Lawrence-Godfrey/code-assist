import fire

from cli import extract, embed, evaluate, generate


class CLI:
    """Main CLI interface for the code assistant package."""

    def __init__(self):
        self.extract = extract.ExtractCommands()
        self.embed = embed.EmbedCommands()
        self.evaluate = evaluate.EvaluateCommands()
        self.generate = generate.GenerateCommands()


def main():
    """Entry point for the CLI."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
